import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
import logging_mp
logger_mp = logging_mp.get_logger(__name__)
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from teleop.utils.config_loader import get_robot_config


class RobotArmIK:
    """
    通用机器人手臂IK解算器
    支持所有在robot_config.yml中配置的机器人类型
    """
    
    def __init__(self, robot_type: str, unit_test: bool = False, visualization: bool = False):
        """
        初始化通用机器人手臂IK解算器
        
        Args:
            robot_type: 机器人类型 (G1_29, G1_23, H1_2, H1, Franka_Panda)
            unit_test: 是否为单元测试模式
            visualization: 是否启用可视化
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        
        self.robot_type = robot_type
        self.unit_test = unit_test
        self.visualization = visualization
        
        # 加载机器人配置
        self.config = get_robot_config(robot_type)
        
        # 根据测试模式选择URDF路径
        if not self.unit_test:
            urdf_path = self.config['urdf_path']
            urdf_dir = self.config['urdf_dir']
        else:
            urdf_path = self.config['urdf_path_test']
            urdf_dir = self.config['urdf_dir_test']
        
        # 构建机器人模型
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, urdf_dir)
        
        # 锁定指定关节
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.config['joints_to_lock'],
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        
        # 添加末端执行器帧
        self.end_effector_ids = {}
        for ee in self.config['end_effectors']:
            parent_joint_id = self.reduced_robot.model.getJointId(ee['parent_joint'])
            offset = np.array(ee['offset'])
            
            self.reduced_robot.model.addFrame(
                pin.Frame(ee['name'],
                          parent_joint_id,
                          pin.SE3(np.eye(3), offset),
                          pin.FrameType.OP_FRAME)
            )
            self.end_effector_ids[ee['name']] = self.reduced_robot.model.getFrameId(ee['name'])
        
        # 创建Casadi模型和数据
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        
        # 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        
        # 创建目标位姿符号变量
        self.target_poses = {}
        for ee_name in self.end_effector_ids.keys():
            self.target_poses[ee_name] = casadi.SX.sym(f"tf_{ee_name.lower()}", 4, 4)
        # 获取手臂关节索引（过滤掉锁定关节）
        self.arm_joint_indices = []
        locked_joints = set(self.config['joints_to_lock'])
        
        # 获取所有未被锁定的关节索引
        available_joint_indices = [i for i, joint_name in enumerate(self.reduced_robot.model.names) 
                                 if joint_name not in locked_joints]
        
        # 只保留在reduced_robot范围内的索引
        self.arm_joint_indices = [i for i in available_joint_indices if i < self.reduced_robot.model.nq]
        
        # 如果没有找到手臂关节，则使用所有可用的关节
        if not self.arm_joint_indices:
            self.arm_joint_indices = list(range(self.reduced_robot.model.nq))
        # 定义误差函数
        self._create_error_functions()
        
        # 定义优化问题
        self._create_optimization_problem()
        
        # 初始化数据
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(
            np.array([0.4, 0.3, 0.2, 0.1]), 
            self.reduced_robot.model.nq
        )
        self.vis = None
        # 初始化可视化
        if self.visualization:
            self._init_visualization()
    
    def _create_error_functions(self):
        """创建误差函数"""
        # 平移误差
        translational_errors = []
        for ee_name, ee_id in self.end_effector_ids.items():
            translational_errors.append(
                self.cdata.oMf[ee_id].translation - self.target_poses[ee_name][:3, 3]
            )
        
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq] + list(self.target_poses.values()),
            [casadi.vertcat(*translational_errors)]
        )
        
        # 旋转误差
        rotational_errors = []
        for ee_name, ee_id in self.end_effector_ids.items():
            rotational_errors.append(
                cpin.log3(self.cdata.oMf[ee_id].rotation @ self.target_poses[ee_name][:3, :3].T)
            )
        
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq] + list(self.target_poses.values()),
            [casadi.vertcat(*rotational_errors)]
        )
    
    def _create_optimization_problem(self):
        """创建优化问题"""
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        
        # 目标位姿参数
        self.param_target_poses = {}
        for ee_name in self.end_effector_ids.keys():
            self.param_target_poses[ee_name] = self.opti.parameter(4, 4)
        
        # 成本函数
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, *list(self.param_target_poses.values()))
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, *list(self.param_target_poses.values()))
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        
        # 设置约束和目标
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit
        ))
        
        # 根据机器人类型调整权重
        is_dual_arm = len(self.config['end_effectors']) > 1
        
        if is_dual_arm:
            # 双臂机器人，平衡位置和姿态
            self.opti.minimize(
                50 * self.translational_cost + 
                0.5 * self.rotation_cost + 
                0.02 * self.regularization_cost + 
                0.1 * self.smooth_cost
            )
        else:
            # 单臂机器人，更注重位置精度
            self.opti.minimize(
                50 * self.translational_cost + 
                self.rotation_cost + 
                0.02 * self.regularization_cost + 
                0.1 * self.smooth_cost
            )
        
        # 求解器设置
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-6
            },
            'print_time': False,
            'calc_lam_p': False
        }
        self.opti.solver("ipopt", opts)
        arm_joint_indices = self.get_arm_joint_indices()
        self._add_collision_constraints(joint3_idx=arm_joint_indices[2], joint2_idx=arm_joint_indices[1], min_height_diff=0.02)  # 桌面安全高度 2cm
    
    def _init_visualization(self):
        """初始化可视化"""
        self.vis = MeshcatVisualizer(
            self.reduced_robot.model, 
            self.reduced_robot.collision_model, 
            self.reduced_robot.visual_model
        )
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        
        # 显示末端执行器帧
        frame_ids = list(self.end_effector_ids.values())
        self.vis.displayFrames(True, frame_ids=frame_ids, axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))
        
        # 添加目标帧可视化
        frame_viz_names = [f'{ee_name}_target' for ee_name in self.end_effector_ids.keys()]
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )
    
    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        if human_left_pose is None:
            robot_right_pose = human_right_pose.copy()
            robot_right_pose[:3, 3] *= scale_factor
            return None, robot_right_pose
        elif human_right_pose is None:
            robot_left_pose = human_left_pose.copy()
            robot_left_pose[:3, 3] *= scale_factor
            return robot_left_pose, None
        else:
            robot_left_pose = human_left_pose.copy()
            robot_right_pose = human_right_pose.copy()
            robot_left_pose[:3, 3] *= scale_factor
            robot_right_pose[:3, 3] *= scale_factor
            return robot_left_pose, robot_right_pose
    
    def solve_ik(self, left_wrist=None, right_wrist=None, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        """
        求解逆运动学
        
        Args:
            left_wrist: 左手腕目标位姿 (4x4矩阵)
            right_wrist: 右手腕目标位姿 (4x4矩阵)
            current_lr_arm_motor_q: 当前关节角度
            current_lr_arm_motor_dq: 当前关节角速度
            
        Returns:
            tuple: (关节角度, 关节力矩)
        """
        if left_wrist is not None and right_wrist is not None:
            target_poses = {'L_ee': left_wrist, 'R_ee': right_wrist}
        elif left_wrist is not None:
            target_poses = {'L_ee': left_wrist}
        elif right_wrist is not None:
            target_poses = {'R_ee': right_wrist}
        else:
            raise ValueError("需要提供left_wrist或right_wrist参数")
        
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)
        
        # 缩放手臂
        left_wrist, right_wrist = self.scale_arms(human_left_pose=left_wrist, human_right_pose=right_wrist)
        
        # 设置可视化目标帧
        if self.visualization:
            for ee_name, pose in target_poses.items():
                self.vis.viewer[f'{ee_name}_target'].set_transform(pose)
        
        # 设置优化参数
        for ee_name, pose in target_poses.items():
            self.opti.set_value(self.param_target_poses[ee_name], pose)
        self.opti.set_value(self.var_q_last, self.init_data)
        try:
            sol = self.opti.solve()
            
            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            
            self.init_data = sol_q
            
            sol_tauff = pin.rnea(
                self.reduced_robot.model, 
                self.reduced_robot.data, 
                sol_q, 
                v, 
                np.zeros(self.reduced_robot.model.nv)
            )
            
            if self.visualization:
                self.vis.display(sol_q)
            
            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")
            
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            
            self.init_data = sol_q
            
            sol_tauff = pin.rnea(
                self.reduced_robot.model, 
                self.reduced_robot.data, 
                sol_q, 
                v, 
                np.zeros(self.reduced_robot.model.nv)
            )
            
            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \ntarget_poses: \n{target_poses}")
            if self.visualization:
                self.vis.display(sol_q)
            
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
    # ----------------------------- 添加碰撞约束 -----------------------------
    def _add_collision_constraints(self, joint3_idx: int, joint2_idx: int, min_height_diff: float = 0.02):
        """
        为关键连杆添加约束，确保肘部关节高于肩部关节
        """
        self.opti.subject_to(self.var_q[joint3_idx] - self.var_q[joint2_idx] >= min_height_diff)

    def get_arm_joint_indices(self):
        """
        获取手臂关节索引
        
        Returns:
            list: 手臂关节索引列表
        """
        return self.arm_joint_indices
    
    def get_solution_joint_indices(self):
        """
        获取解决方案对应的关节索引（用于仿真控制）
        
        Returns:
            list: 解决方案关节索引列表，长度与sol_q相同
        """
        # 返回从0开始的连续索引，长度与reduced_robot的关节数相同
        return list(range(self.reduced_robot.model.nq))
    
    def get_joint_names(self):
        """
        获取关节名称列表
        
        Returns:
            list: 关节名称列表
        """
        return self.reduced_robot.model.names
    
    def get_locked_joints(self):
        """
        获取锁定关节列表
        
        Returns:
            list: 锁定关节名称列表
        """
        return self.config['joints_to_lock']
    
    def get_available_joints(self):
        """
        获取可用关节（未被锁定）列表
        
        Returns:
            list: 可用关节名称列表
        """
        locked_joints = set(self.config['joints_to_lock'])
        return [joint_name for joint_name in self.reduced_robot.model.names 
                if joint_name not in locked_joints]


# 兼容性函数，保持与原有接口一致
def create_arm_ik(robot_type: str, unit_test: bool = False, visualization: bool = False):
    """
    创建机器人手臂IK解算器的工厂函数
    
    Args:
        robot_type: 机器人类型
        unit_test: 是否为单元测试模式
        visualization: 是否启用可视化
        
    Returns:
        UniversalRobotArmIK: IK解算器实例
    """
    return UniversalRobotArmIK(robot_type, unit_test, visualization)


if __name__ == "__main__":
    # 测试代码
    import time
    
    # 测试不同类型的机器人
    robot_types = ['Franka_Panda']
    
    for robot_type in robot_types:
        print(f"\n测试 {robot_type} 机器人...")
        
        try:
            arm_ik = UniversalRobotArmIK(robot_type, unit_test=True, visualization=True)
            
            # 从配置文件判断是否是双臂机器人
            is_dual_arm = len(arm_ik.config['end_effectors']) > 1
            
            # 创建测试目标位姿
            if is_dual_arm:
                # 双臂机器人
                left_wrist = pin.SE3(
                    pin.Quaternion(1, 0, 0, 0),
                    np.array([0.3, 0.2, 0.5])
                ).homogeneous
                right_wrist = pin.SE3(
                    pin.Quaternion(1, 0, 0, 0),
                    np.array([0.3, -0.2, 0.5])
                ).homogeneous
                
                # 求解IK
                sol_q, sol_tauff = arm_ik.solve_ik(left_wrist=left_wrist, right_wrist=right_wrist)
            else:
                # 单臂机器人
                right_wrist = pin.SE3(
                    pin.Quaternion(1, 0, 0, 0),
                    np.array([0.5, 0.0, 0.6])
                ).homogeneous
                
                # 求解IK
                sol_q, sol_tauff = arm_ik.solve_ik(right_wrist=right_wrist)
            
            print(f"{robot_type} IK求解成功，关节角度: {sol_q}")
            print(f"手臂关节索引: {arm_ik.arm_joint_indices}")
            print(f"解决方案关节索引: {arm_ik.get_solution_joint_indices()}")
            print(f"锁定关节: {arm_ik.get_locked_joints()}")
            print(f"可用关节: {arm_ik.get_available_joints()}")
            
            # 等待一段时间以便观察可视化
            time.sleep(2)
            
        except Exception as e:
            print(f"{robot_type} 测试失败: {e}")
    
    print("\n所有测试完成！")
