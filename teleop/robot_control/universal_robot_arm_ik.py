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


class UniversalRobotArmIK:
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
        self.joints_to_lock = self.config['joints_to_lock']
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.joints_to_lock,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        
        # 添加末端执行器帧
        self.end_effector_ids = {}
        for ee in self.config['end_effectors']:
            end_joint_id = self.reduced_robot.model.getJointId(ee['end_joint'])
            offset = np.array(ee['offset'])
            
            self.reduced_robot.model.addFrame(
                pin.Frame(ee['name'],
                          end_joint_id,
                          pin.SE3(np.eye(3), offset),
                          pin.FrameType.OP_FRAME)
            )
            self.end_effector_ids[ee['name']] = self.reduced_robot.model.getFrameId(ee['name'])
        
        # 获取末端关节和链接ID（用于正运动学）
        if self.config['end_effectors']:
            first_ee = self.config['end_effectors'][0]
            self.end_joint_id = self.reduced_robot.model.getJointId(first_ee['end_joint'])
            self.end_link_id = self.reduced_robot.model.getFrameId(first_ee['end_link'])
        
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
        # 获取缩放因子
        self.robot_arm_length = self.config['arm_scaling']['robot_length'] 
        self.human_arm_length = self.config['arm_scaling']['human_length']

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
        
        # 成本函数 - 使用已经创建的函数
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
        arm_joint_indices = self.get_arm_joint_indices()
        
        # 添加高度约束成本（如果适用）
        height_constraint_cost = 0
        if len(arm_joint_indices) >= 5:
            height_constraint_cost = self._add_collision_constraints(
                frame_idx_low=arm_joint_indices[2], 
                frame_idx_high=arm_joint_indices[3], 
                min_height_diff=0.02,
                penalty_weight=50.0
            )
        
        # 根据机器人类型调整权重
        is_dual_arm = len(self.config['end_effectors']) > 1
        
        if is_dual_arm:
            # 双臂机器人，平衡位置和姿态
            self.opti.minimize(
                50 * self.translational_cost + 
                0.5 * self.rotation_cost + 
                0.02 * self.regularization_cost + 
                0.1 * self.smooth_cost +
                height_constraint_cost
            )
        else:
            # 单臂机器人，更注重位置精度
            self.opti.minimize(
                50 * self.translational_cost + 
                self.rotation_cost + 
                0.02 * self.regularization_cost + 
                0.1 * self.smooth_cost +
                height_constraint_cost
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
    
    # 如果机械臂尺寸与你的手臂尺寸不一致 :)
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
        left_wrist, right_wrist = self.scale_arms(human_left_pose=left_wrist, human_right_pose=right_wrist, human_arm_length=self.config['arm_scaling']['human_length'], robot_arm_length=self.config['arm_scaling']['robot_length'])
        
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
    def _add_collision_constraints(self, frame_idx_low: int, frame_idx_high: int, min_height_diff: float = 0.02, penalty_weight: float = 5.0):
        """
        添加高度约束：确保 frame_idx_high 的 Z 坐标始终比 frame_idx_low 高 min_height_diff
        
        Args:
            frame_idx_low: 较低关节的帧索引
            frame_idx_high: 较高关节的帧索引
            min_height_diff: 最小高度差（米）
            penalty_weight: 惩罚权重
            
        Returns:
            float: 高度约束成本
        """
        # 创建高度约束函数 - 使用SX符号变量
        q_sx = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q_sx)
        
        # 获取关节位置
        frame_low_pos = self.cdata.oMf[frame_idx_low].translation
        frame_high_pos = self.cdata.oMf[frame_idx_high].translation
        
        # 计算高度差
        height_diff = frame_high_pos[2] - frame_low_pos[2]
        
        # 创建高度约束函数
        height_constraint_func = casadi.Function(
            "height_constraint",
            [q_sx],
            [height_diff - min_height_diff]
        )
        
        # 应用约束到优化变量并返回成本
        height_violation = casadi.fmax(0, min_height_diff - height_constraint_func(self.var_q))
        return penalty_weight * casadi.sumsqr(height_violation)


    def _add_end_effector_link_collision_constraints(self):
        """
        添加末端执行器与连杆的碰撞约束
        确保末端执行器不会与机器人的连杆发生碰撞
        """
        # 获取手臂关节索引
        arm_joint_indices = self.get_arm_joint_indices()
        
        # 根据机器人类型设置不同的安全距离
        if self.robot_type in ['G1_29', 'G1_23', 'H1_2', 'H1']:
            # 人形机器人，需要更大的安全距离
            min_safe_distance = 0.08  # 8cm
        elif self.robot_type in ['Franka_Panda']:
            # Franka机器人，中等安全距离
            min_safe_distance = 0.06  # 6cm
        else:
            # 其他机器人，默认安全距离
            min_safe_distance = 0.05  # 5cm
        
        # 创建碰撞惩罚成本函数
        collision_cost = 0
        
        # 为每个末端执行器添加碰撞惩罚
        for ee_name, ee_id in self.end_effector_ids.items():
            # 获取末端执行器的位姿函数
            ee_pose = self.cdata.oMf[ee_id]
            ee_position = ee_pose.translation
            
            # 为关键连杆添加距离惩罚
            for i, joint_idx in enumerate(arm_joint_indices[:-2]):  # 跳过最后两个关节
                # 获取连杆的位姿
                link_pose = self.cdata.oMf[joint_idx]
                link_position = link_pose.translation
                
                # 计算末端执行器与连杆之间的距离
                distance = casadi.norm_2(ee_position - link_position)
                
                # 添加距离惩罚：当距离小于安全距离时增加惩罚
                safe_distance_penalty = casadi.fmax(0, min_safe_distance - distance)
                collision_cost += 10.0 * casadi.sumsqr(safe_distance_penalty)
                
                # 添加高度惩罚：确保末端执行器不会低于关键连杆
                if i < 2:  # 只对前两个连杆添加高度约束
                    height_diff = ee_position[2] - link_position[2]
                    height_penalty = casadi.fmax(0, -height_diff - 0.02)  # 允许2cm的误差
                    collision_cost += 5.0 * casadi.sumsqr(height_penalty)
                
                # 限制约束数量，避免过度约束
                if i >= 2:  # 只对前3个连杆添加约束
                    break
        
        # 将碰撞成本添加到总成本函数中
        self.collision_cost = collision_cost

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
    
    def forward_kinematics(self, joint_angles) -> np.ndarray:
        """
        计算前向运动学
        
        Args:
            joint_angles: 关节角度数组或列表 (nq,)
            
        Returns:
            np.ndarray: 末端执行器的4x4变换矩阵
        """
        # 确保输入是numpy数组
        if not isinstance(joint_angles, np.ndarray):
            joint_angles = np.array(joint_angles, dtype=np.float64)
        
        if len(joint_angles) != self.reduced_robot.model.nq:
            raise ValueError(f"关节角度数量不匹配: 期望 {self.reduced_robot.model.nq}, 实际 {len(joint_angles)}")
        
        # 检查末端执行器配置
        if not self.end_effector_ids:
            raise ValueError("没有配置末端执行器")
        
        try:
            # 使用Pinocchio的forwardKinematics计算正运动学
            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, joint_angles)
            pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
            
            # 获取末端执行器的位姿
            pose = self.reduced_robot.data.oMf[self.end_link_id]
            
            # 返回4x4变换矩阵
            return pose.homogeneous
            
        except Exception as e:
            raise ValueError(f"无法计算末端执行器 {self.end_link_id} 的正运动学: {e}")
    
    def forward_kinematics_position_quaternion(self, joint_angles):
        """
        计算前向运动学，返回位置和四元数
        
        Args:
            joint_angles: 关节角度数组或列表 (nq,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (位置 [x, y, z], 四元数 [w, x, y, z])
        """
        # 确保输入是numpy数组
        if not isinstance(joint_angles, np.ndarray):
            joint_angles = np.array(joint_angles, dtype=np.float64)
        
        if len(joint_angles) != self.reduced_robot.model.nq:
            raise ValueError(f"关节角度数量不匹配: 期望 {self.reduced_robot.model.nq}, 实际 {len(joint_angles)}")
        
        # 检查末端执行器配置
        if not self.end_effector_ids:
            raise ValueError("没有配置末端执行器")
        
        # 获取第一个末端执行器的名称
        ee_names = list(self.end_effector_ids.keys())
        if not ee_names:
            raise ValueError("末端执行器名称列表为空")
        
        ee_name = ee_names[0]  # 获取第一个末端执行器
        ee_id = self.end_effector_ids[ee_name]
        
        try:
            # 使用Pinocchio的forwardKinematics计算正运动学
            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, joint_angles)
            pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
            
            # 获取末端执行器的位姿
            pose = self.reduced_robot.data.oMf[ee_id]
            
            # 提取位置和旋转矩阵
            position = pose.translation
            rotation_matrix = pose.rotation
            
            # 转换为四元数
            quat = pin.Quaternion(rotation_matrix)
            quaternion = np.array([quat.w, quat.x, quat.y, quat.z])
            
            return position, quaternion
            
        except Exception as e:
            raise ValueError(f"无法计算末端执行器 {ee_name} 的正运动学: {e}")
    
    def compute_jacobian(self, joint_angles) -> np.ndarray:
        """
        计算雅可比矩阵
        
        Args:
            joint_angles: 关节角度数组或列表 (nq,)
            
        Returns:
            np.ndarray: 雅可比矩阵 (6, n_joints)
        """
        # 确保输入是numpy数组
        if not isinstance(joint_angles, np.ndarray):
            joint_angles = np.array(joint_angles, dtype=np.float64)
        
        if len(joint_angles) != self.reduced_robot.model.nq:
            raise ValueError(f"关节角度数量不匹配: 期望 {self.reduced_robot.model.nq}, 实际 {len(joint_angles)}")
        
        # 检查末端执行器配置
        if not self.end_effector_ids:
            raise ValueError("没有配置末端执行器")
        
        # 获取第一个末端执行器的名称
        ee_names = list(self.end_effector_ids.keys())
        if not ee_names:
            raise ValueError("末端执行器名称列表为空")
        
        ee_name = ee_names[0]  # 获取第一个末端执行器
        ee_id = self.end_effector_ids[ee_name]
        
        try:
            # 计算雅可比矩阵
            pin.computeJointJacobians(self.reduced_robot.model, self.reduced_robot.data, joint_angles)
            jacobian = pin.getJointJacobian(
                self.reduced_robot.model, 
                self.reduced_robot.data, 
                joint_angles, 
                ee_id, 
                pin.ReferenceFrame.WORLD
            )
            
            return jacobian[:6, :]  # 返回位置和姿态的雅可比矩阵
            
        except Exception as e:
            raise ValueError(f"无法计算末端执行器 {ee_name} 的雅可比矩阵: {e}")
    
    def inverse_kinematics(self, target_pose: np.ndarray, initial_guess: np.ndarray = None) -> np.ndarray:
        """
        计算逆运动学
        
        Args:
            target_pose: 目标位姿矩阵 (4x4)
            initial_guess: 初始关节角度猜测 (可选)
            
        Returns:
            np.ndarray: 关节角度解
        """
        # 检查末端执行器配置
        if not self.end_effector_ids:
            raise ValueError("没有配置末端执行器")
        
        # 确定目标位姿对应的末端执行器
        ee_names = list(self.end_effector_ids.keys())
        if not ee_names:
            raise ValueError("末端执行器名称列表为空")
        
        ee_name = ee_names[0]  # 获取第一个末端执行器
        
        # 设置初始猜测
        if initial_guess is not None:
            self.init_data = initial_guess
        else:
            self.init_data = np.zeros(self.reduced_robot.model.nq)
        
        # 调用现有的solve_ik方法
        if ee_name == 'L_ee':
            sol_q, _ = self.solve_ik(left_wrist=target_pose)
        else:
            sol_q, _ = self.solve_ik(right_wrist=target_pose)
        
        return sol_q


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
