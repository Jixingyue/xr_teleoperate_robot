#!/usr/bin/env python3
"""
PyBullet仿真脚本 - 简单平滑版本（使用四元数球面插值）
基于原始teleop_hand_and_arm.py，但使用PyBullet替代Isaac Gym
"""

from numpy.random import f
import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import sys
import os
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation as R

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from televuer import TeleVuerWrapper
from robot_control.universal_robot_arm_ik import UniversalRobotArmIK as ArmIK
# 添加性能监控模块
from utils.performance_monitor import PerformanceMonitor, PerformanceAnalyzer
class PyBulletSimSimpleSmooth:
    def __init__(self, print_freq=False, enable_performance_monitoring=True):
        self.print_freq = print_freq
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # 初始化性能监控器
        if self.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(name="PyBulletSim", window_size=100)
            self.frame_count = 0
            self.last_performance_log = time.time()
            self.performance_log_interval = 5.0  # 每5秒输出一次性能统计
        self.arm_ik = ArmIK(robot_type='Franka_Panda', unit_test=False, visualization=False)
        # 获取夹爪关节索引
        # 获取夹爪关节索引，配置文件只有夹爪关节名称，没有索引
        self.gripper_joint = self.arm_ik.config['gripper_joints']
        self.gripper_joint_indices = [joint["index"] for joint in self.gripper_joint]
        # 初始化PyBullet - 优化性能设置
        self.physics_client = p.connect(p.GUI)  # 或者使用 p.DIRECT 进行无GUI仿真
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/60.0)
        
        # 性能优化设置
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # 禁用RGB预览
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # 禁用深度预览
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # 禁用分割预览
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 尝试加载桌子（如果存在）
        try:
            table_urdf = "table/table.urdf"
            table_pos = [0, 0, 0.0]
            table_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.table_id = p.loadURDF(table_urdf, table_pos, table_orientation)
        except:
            print("警告: 无法加载桌子模型，将使用立方体代替")
            # 创建一个简单的桌子
            table_size = [0.8, 0.8, 0.1]
            table_pos = [0, 0, 1.1]
            self.table_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=table_size),
                basePosition=table_pos
            )
        
        # 尝试加载立方体（如果存在）
        try:
            cube_pos = [0, 0, 1.25]
            cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.cube_id = p.loadURDF("cube_small.urdf", cube_pos, cube_orientation)
        except:
            print("警告: 无法加载立方体模型，将创建一个简单的立方体")
            # 创建一个简单的立方体
            cube_size = [0.025, 0.025, 0.025]
            cube_pos = [0, 0, 1.25]
            self.cube_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_size),
                basePosition=cube_pos
            )
        
        # 尝试加载机械臂
        try:
            arm_urdf = "assets/franka_description/panda_with_gripper.urdf"
            arm_pos = [0, 0, 0.6]  # 将机械臂基座放在桌子上方0.01米
            arm_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.arm_id = p.loadURDF(arm_urdf, arm_pos, arm_orientation, useFixedBase=True)
        except:
            print("警告: 无法加载机械臂模型模型，将创建一个简单的机械臂")
            # 创建一个简单的机械臂模型
            raise Exception("无法加载机械臂模型模型")
        
        # 获取机械臂关节信息
        self.num_joints = p.getNumJoints(self.arm_id)
        self.joint_names = []
        self.joint_indices = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.arm_id, i)
            self.joint_names.append(joint_info[1].decode('utf-8'))
            if joint_info[2] == p.JOINT_REVOLUTE:  # 只考虑旋转关节
                self.joint_indices.append(i)
        self.end_effector_index = self.joint_indices[-1]
        print(f"机械臂关节数量: {len(self.joint_indices)}")
        print(f"关节名称: {[self.joint_names[i] for i in self.joint_indices]}")
        print(f"机械臂基座位置: {arm_pos}")
        print(f"桌子高度: 0.6米")
        self.target_q = np.zeros(len(self.joint_indices))
        # 设置相机参数
        self.cam_distance = 2.0
        self.cam_yaw = 50
        self.cam_pitch = -35
        self.cam_target_pos = [0, 0, 0.8]  # 调整相机目标到桌子高度
        
        # 相机偏移参数
        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-2.5, 0, 1.4])  # 调整相机位置，更好地观察桌子
        
        # 性能优化参数
        self.camera_resolution = (640, 480)  # 降低分辨率以提高性能
        self.camera_fov = 60
        self.camera_near = 0.1
        self.camera_far = 10.0
        self.skip_camera_rendering = False  # 可选的相机渲染跳过
        self.camera_render_interval = 1  # 每N帧渲染一次相机图像
        
        # 自适应性能调整参数
        self.performance_threshold = 20.0  # 20ms阈值
        self.adaptive_adjustment = True  # 启用自适应调整
        self.last_adjustment_time = time.time()
        self.adjustment_interval = 2.0  # 每2秒调整一次
        
        # 初始化平滑参数
        self.prev_head_rmat = None
        self.smooth_factor = 0.15  # 平滑因子，可调节
        
        # 设置相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=self.cam_distance,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=self.cam_target_pos
        )
        #设置初始化关节角度
        self.init_joint_angles = np.deg2rad(np.array([0,-36.13,130.63,164.40,82.87,5.0]))   
        for i, angle in zip(self.joint_indices, self.init_joint_angles):
            p.resetJointState(self.arm_id, i, angle)
        #打印初始化关节信息，角度、末端执行器位姿
        print(f"初始化关节角度: {self.init_joint_angles}")
        print(f"初始化末端执行器位姿: {self.get_current_state()['end_transform_matrix']}")

    
    def render_camera_images(self):
        """渲染双目相机图像"""
        # 使用固定相机位置
        left_cam_pos = self.cam_pos + self.left_cam_offset
        right_cam_pos = self.cam_pos + self.right_cam_offset
        left_cam_target = left_cam_pos + self.cam_lookat_offset
        right_cam_target = right_cam_pos + self.cam_lookat_offset
        
        # 监控相机图像获取性能
        if self.enable_performance_monitoring:
            self.performance_monitor.start_timer("camera_rendering")
        
        # 优化相机渲染：降低分辨率，使用更快的渲染器
        # 获取相机图像
        # 模拟左眼图像
        left_view_matrix = p.computeViewMatrix(left_cam_pos, left_cam_target, [0, 0, 1])
        left_projection_matrix = p.computeProjectionMatrixFOV(
            self.camera_fov, 
            self.camera_resolution[0] / self.camera_resolution[1], 
            self.camera_near, 
            self.camera_far
        )
        left_image = p.getCameraImage(
            self.camera_resolution[0], 
            self.camera_resolution[1], 
            left_view_matrix, 
            left_projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        
        # 模拟右眼图像
        right_view_matrix = p.computeViewMatrix(right_cam_pos, right_cam_target, [0, 0, 1])
        right_projection_matrix = p.computeProjectionMatrixFOV(
            self.camera_fov, 
            self.camera_resolution[0] / self.camera_resolution[1], 
            self.camera_near, 
            self.camera_far
        )
        right_image = p.getCameraImage(
            self.camera_resolution[0], 
            self.camera_resolution[1], 
            right_view_matrix, 
            right_projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        
        # 提取RGB图像数据
        left_rgb = left_image[2][:, :, :3]  # RGB数据，直接切片
        right_rgb = right_image[2][:, :, :3]  # RGB数据，直接切片
        
        if self.enable_performance_monitoring:
            self.performance_monitor.end_timer("camera_rendering")
            
        return left_rgb, right_rgb
    def solve_inverse_kinematics(self, right_pose, current_lr_arm_motor_q=None):
        """执行逆运动学计算"""
        # 监控逆运动学计算性能
        if self.enable_performance_monitoring:
            self.performance_monitor.start_timer("inverse_kinematics")
            
        if self.frame_count % 60 == 0:  # 每60帧输出一次
            print(f"执行位姿，帧 {self.frame_count}: right_pose=\n{right_pose}")  
            if current_lr_arm_motor_q is not None:
                print(f"当前关节角度: {current_lr_arm_motor_q}...")
        
        # 使用当前关节角度作为初始猜测，提高逆运动学求解效率
        if current_lr_arm_motor_q is not None:
            sol_q, _ = self.arm_ik.solve_ik(right_wrist=right_pose, current_lr_arm_motor_q=current_lr_arm_motor_q)
        else:
            sol_q, _ = self.arm_ik.solve_ik(right_wrist=right_pose)            
        if self.frame_count % 60 == 0:  # 每60帧输出一次
            print(f"逆运动学求解结果: {sol_q}")
            
        # 如果求解失败，使用当前关节角度
        if sol_q is None and current_lr_arm_motor_q is not None:
            print("逆运动学求解失败，使用当前关节角度")
            sol_q = current_lr_arm_motor_q
        elif sol_q is None:
            print("逆运动学求解失败且无当前关节角度，跳过此帧")
            if self.enable_performance_monitoring:
                self.performance_monitor.end_timer("inverse_kinematics")
            return None
            
        # 检查关节角度是否在合理范围内
        if self.frame_count % 60 == 0:  # 每60帧检查一次
            for i, (q, (min_limit, max_limit)) in enumerate(zip(sol_q, self.arm_ik.config['joint_limits'])):
                if q < min_limit or q > max_limit:
                    print(f"警告: 关节{i+1}角度{q:.3f}超出限制[{min_limit:.3f}, {max_limit:.3f}]")
                    # 限制关节角度
                    sol_q[i] = np.clip(q, min_limit, max_limit)
                    
        if self.enable_performance_monitoring:
            self.performance_monitor.end_timer("inverse_kinematics")
            
        return sol_q

    def control_joints(self, target_angles, right_trigger):
        """控制机械臂关节和夹爪"""
        # 设置关节角度
        self.frame_count += 1
        for i, angle in zip(self.joint_indices, target_angles):
            # p.resetJointState(self.arm_id, i, angle)
            p.setJointMotorControl2(
                    self.arm_id, 
                    i, 
                    p.POSITION_CONTROL, 
                    targetPosition=angle,
                    force=100.0,  # 增加力矩以确保关节能够到达目标位置
                    maxVelocity=30.0  # 限制最大速度
                )
        # 设置夹爪状态
        if self.gripper_joint_indices is not None and len(self.gripper_joint_indices) > 0:
            joint_value= right_trigger/100.0/2.0
            joint_value =0.04 if joint_value >= 0.04 else joint_value
            for i, joint_idx in enumerate(self.gripper_joint_indices):
                p.resetJointState(self.arm_id, joint_idx,joint_value ) # franka夹爪配置
        # 等待仿真稳定，修改为角度差小于0.05时停止，最高等待60帧
        max_frames = 0
        while True:
            p.stepSimulation()
            max_frames += 1
            time.sleep(1/240.0)
            current_angles = self.get_current_state()['joint_angles']
            if np.allclose(target_angles, current_angles, atol=0.5):
                break
            if max_frames > 60:
                break

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos, right_trigger, left_trigger, current_lr_arm_motor_q=None,current_pos=None,current_ori=None):
        # 开始整体性能监控
        if self.enable_performance_monitoring:
            self.performance_monitor.start_timer("total_step")
            self.frame_count += 1
        
        if self.print_freq:
            start = time.time()
        
        # 渲染相机图像
        left_rgb, right_rgb = self.render_camera_images()
        
        # 执行逆运动学计算
        sol_q = self.solve_inverse_kinematics(right_pose, current_lr_arm_motor_q)
        # sol_q = self.inverse_kin(right_pose,current_lr_arm_motor_q)
        if sol_q is None:
            print("逆运动学求解失败，跳过此帧")
            return left_rgb, right_rgb
        # 执行关节控制
        self.target_q = sol_q
        self.control_joints(sol_q, right_trigger)
        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))
        
        return left_rgb, right_rgb
    
    def get_current_state(self) -> dict:
        """获取当前机械臂状态（参考simulation_test.py的实现）"""
        try:
            # 检查机械臂ID是否有效
            if not hasattr(self, 'arm_id') or self.arm_id is None:
                raise ValueError("机械臂ID无效")
            
            # 获取关节角度
            joint_states = p.getJointStates(self.arm_id, self.joint_indices)
            joint_angles = [state[0] for state in joint_states]
            
            # 获取末端执行器位姿
            end_effector_pose = p.getLinkState(self.arm_id, self.end_effector_index)
            end_pos = list(end_effector_pose[0])  # 转换为列表以便修改
            end_quat = end_effector_pose[1]
            
            # 构建4x4变换矩阵
            end_rot_matrix = p.getMatrixFromQuaternion(end_quat)
            end_rot_matrix = np.array(end_rot_matrix).reshape(3, 3)
            
            end_transform_matrix = np.eye(4)
            end_transform_matrix[:3, :3] = end_rot_matrix
            end_transform_matrix[:3, 3] = end_pos
            # 坐标系调整：Z轴减去基座高度
            base_height = 0.8  # PyBullet中机械臂基座高度
            end_pos[2] -= base_height
            end_transform_matrix[2, 3] -= base_height
            return {
                'joint_angles': joint_angles,
                'end_position': end_pos,
                'end_orientation': end_quat,
                'end_transform_matrix': end_transform_matrix
            }
        except Exception as e:
            print(f"获取机械臂状态失败: {e}")
            print(f"机械臂ID: {getattr(self, 'arm_id', 'None')}")
            print(f"关节索引: {getattr(self, 'joint_indices', 'None')}")
            print(f"末端执行器索引: {getattr(self, 'end_effector_index', 'None')}")
            raise e
        
    def print_state(self, state: dict):
        """打印机械臂状态（参考simulation_test.py的实现）"""
        print("\n=== 机械臂当前状态 ===")
        print(f"关节角度 (弧度): {[f'{angle:.4f}' for angle in state['joint_angles']]}")
        print(f"关节角度 (度): {[f'{np.degrees(angle):.2f}' for angle in state['joint_angles']]}")
        print("末端4x4变换矩阵:")
        print(state['end_transform_matrix'])
        print("=====================\n")
    
    def end(self):
        p.disconnect()

if __name__ == '__main__':
    # 配置参数（参考原始teleop_hand_and_arm.py）
    BINOCULAR = False  # 双目模式
    xr_mode = "control"  # 手部追踪模式 "hand" 或 "control"
    
    # 图像配置
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 640],  # 头部相机分辨率
        'head_camera_id_numbers': [0,1],
    }
    
    # 计算图像形状
    ASPECT_RATIO_THRESHOLD = 2.0
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
    
    # 创建共享内存
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
    
    # 初始化television wrapper（参考原始代码）
    tv_wrapper = TeleVuerWrapper(
        binocular=BINOCULAR, 
        use_hand_tracking=xr_mode == "hand", 
        img_shape=tv_img_shape, 
        img_shm_name=tv_img_shm.name, 
        return_state_data=True, 
        return_hand_rot_data=False,
        robot_type="AUBO_I5"  # 指定机器人类型为UR3E
    )
    
    # 初始化仿真器
    simulator = PyBulletSimSimpleSmooth()

    try:
        print("PyBullet仿真启动（简单平滑版本），按Ctrl+C退出...")
        print("性能监控已启用，将每5秒输出一次性能统计")
        
        # 创建主循环性能监控器
        main_monitor = PerformanceMonitor(name="MainLoop", window_size=100)
        
        while True:
            # 监控主循环性能
            main_monitor.start_timer("main_loop_iteration")
            
            # 获取遥操作数据
            main_monitor.start_timer("teleop_data_acquisition")
            tele_data = tv_wrapper.get_motion_state_data()
            main_monitor.end_timer("teleop_data_acquisition")
            
            # 提取头部和手部数据
            main_monitor.start_timer("data_processing")
            head_rmat = tele_data.head_rotation_matrix if hasattr(tele_data, 'head_rotation_matrix') else np.eye(3)
            left_pose = tele_data.left_arm_pose if hasattr(tele_data, 'left_arm_pose') else np.eye(4)
            right_pose = tele_data.right_arm_pose if hasattr(tele_data, 'right_arm_pose') else np.eye(4)
            left_qpos = None
            right_qpos = None
            right_trigger = tele_data.right_trigger_value if hasattr(tele_data, 'right_trigger_value') else False
            left_trigger = tele_data.left_trigger_value if hasattr(tele_data, 'left_trigger_value') else False
            main_monitor.end_timer("data_processing")

            # 执行仿真步进
            main_monitor.start_timer("simulation_step")
            current_state = simulator.get_current_state()
            current_lr_arm_motor_q = current_state['joint_angles']
            current_pos=current_state['end_position']
            current_ori=current_state['end_orientation']
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos, right_trigger, left_trigger,current_lr_arm_motor_q,current_pos,current_ori)
            main_monitor.end_timer("simulation_step")
            
            # 获取更新后的机械臂状态（用于验证和调试）
            end_transform_matrix = None
            if hasattr(simulator, 'arm_id') and simulator.arm_id is not None:
                try:
                    end_transform_matrix = current_state['end_transform_matrix']
                    # 验证变换矩阵的有效性
                    if simulator.frame_count % 60 == 0:  # 每60帧验证一次
                        # 检查旋转矩阵的正交性
                        end_rot_matrix = end_transform_matrix[:3, :3]
                        rot_valid = np.allclose(np.dot(end_rot_matrix, end_rot_matrix.T), np.eye(3), atol=1e-6)
                        det_valid = np.abs(np.linalg.det(end_rot_matrix) - 1.0) < 1e-6
                        print(f"旋转矩阵验证 - 正交性: {rot_valid}, 行列式: {det_valid}")
                        
                        # 计算位置误差
                        target_pos = right_pose[:3, 3]
                        actual_pos = np.array(current_state['end_position'])
                        #ik解算时存在缩放因子
                        scale_factor = simulator.arm_ik.robot_arm_length /  simulator.arm_ik.human_arm_length
                        scale_target_pos = target_pos * scale_factor
                        pos_error = np.linalg.norm(actual_pos - scale_target_pos)
                        print(f"目标位置: {target_pos}，缩放后的目标位置: {target_pos * scale_factor}")
                        print(f"实际位置: {actual_pos}")
                        print(f"位置误差: {pos_error:.4f} 米") 
                        print(f"目标关节角度：{simulator.target_q}")
                        print(f"当前关节角度: {current_state['joint_angles']}")

                        
                        # 计算旋转误差
                        target_rot = right_pose[:3, :3]
                        actual_rot = end_rot_matrix
                        rot_error = np.linalg.norm(target_rot - actual_rot)
                        print(f"旋转误差: {rot_error:.4f}")
                        
                        # 检查误差是否在可接受范围内
                        if pos_error > 0.01:  # 1cm误差阈值
                            print("警告: 位置误差过大!")
                        if rot_error > 0.1:  # 旋转误差阈值
                            print("警告: 旋转误差过大!")
                            
                except Exception as e:
                    print(f"获取更新后机械臂状态失败: {e}")
                    end_transform_matrix = None
            
            # 调试输出（可选）- 减少输出频率以提高性能
            if simulator.frame_count % 60 == 0:  # 每60帧输出一次
                print(f"帧 {simulator.frame_count}: head_rmat={head_rmat}, right_pose={right_pose}, trigger={right_trigger}")
                if end_transform_matrix is not None:
                    simulator.print_state(current_state)
            
            # 合并双目图像
            main_monitor.start_timer("image_merging")
            head_image = np.hstack((left_img, right_img))
            np.copyto(tv_img_array, head_image)
            main_monitor.end_timer("image_merging")
            
            # 结束主循环迭代计时
            main_monitor.end_timer("main_loop_iteration")
            
            # 定期输出主循环性能统计
            if simulator.frame_count % 150 == 0:  # 每150帧（约5秒）输出一次
                main_monitor.log_performance("主循环性能")
            
            # 自适应延迟控制
            main_loop_time = main_monitor.get_stats("main_loop_iteration")
            if main_loop_time and main_loop_time['latest'] < 16.0:  # 如果主循环时间小于16ms
                time.sleep(0.016 - main_loop_time['latest'] / 1000.0)  # 动态调整延迟
            else:
                time.sleep(0.001)  # 最小延迟
            
    except KeyboardInterrupt:
        print("正在退出仿真...") 
        simulator.end()
        tv_img_shm.close()
        tv_img_shm.unlink()        
        exit(0)