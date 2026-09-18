#!/usr/bin/env python3
"""
通用机械臂控制器具体实现
包含UR3E和Franka Panda的具体实现
"""

import numpy as np
import threading
import time
import socket
import struct
import logging_mp
from robot_control.robot_arm_basic import (
    RobotArmController, MotorState, LowState, DataBuffer
)
from .aubo_sdk import Auboi5Robot,RobotErrorType
logger_mp = logging_mp.get_logger(__name__)

# ============================================================================
# 具体实现类
# ============================================================================

class UR3E_ArmController(RobotArmController):
    """
    UR3E机械臂控制器
    继承自RobotArmController抽象基类
    """
    
    def __init__(self, robot_ip: str = "192.168.1.100", robot_port: int = 30003, 
                 motion_mode: bool = False):
        """
        初始化UR3E机械臂控制器
        
        Args:
            robot_ip: 机器人IP地址
            robot_port: 机器人端口（命令发送端口）
            motion_mode: 是否为运动模式
        """
        super().__init__("UR3E", motion_mode)
        
        # UR3E特定参数
        self.robot_ip = robot_ip
        self.robot_port = robot_port  # UR机器人端口（状态获取和命令发送都使用此端口）                
        
    def _init_robot_connection(self) -> bool:
        """
        初始化UR3E机器人连接
        """
        try:
            logger_mp.info(f"连接到UR3E机器人: {self.robot_ip}:{self.robot_port}")
            self.robot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.robot.connect((self.robot_ip, self.robot_port))
            # 测试连接
            test_pos = self._get_current_pos()
            if test_pos is not None:
                logger_mp.info(f"UR3E机器人连接成功，当前关节角度: {test_pos}")
                return True
            else:
                logger_mp.error("无法获取机器人状态，连接失败")
                return False
            
        except Exception as e:
            logger_mp.error(f"连接UR3E机器人失败: {e}")
            return False
    
    def _subscribe_motor_state(self):
        """
        订阅电机状态数据
        """
        while self._subscribe_running:
            # 通过socket获取真实关节状态
            current_pos = self._get_current_pos()
            if current_pos is not None:
                self.current_joint_angles = current_pos
                # 计算关节速度（简单差分）
                if hasattr(self, '_last_positions'):
                    self.current_joint_velocities = (current_pos - self._last_positions) / self.control_dt
                self._last_positions = current_pos.copy()
            
            # 更新数据缓冲区
            # with self.data_lock:
            # 创建统一状态对象
            state = LowState(self.num_joints)  # UR3E有6个关节
            
            # 更新电机状态
            for i, motor_state in enumerate(state.motor_state):
                motor_state.q = self.current_joint_angles[i] if i < len(self.current_joint_angles) else 0.0
                motor_state.dq = self.current_joint_velocities[i] if i < len(self.current_joint_velocities) else 0.0
            
            # 保存到数据缓冲区
            self.state_buffer.SetData(state)
            
            time.sleep(self.control_dt)  # 500Hz更新频率
    
    def _publish_motor_command(self):
        """
        发布电机控制命令
        """
        if self.motion_mode:
            pass
        while self._publish_running:
            try:
                start_time = time.time()
                
                with self.ctrl_lock:
                    q_target = self.q_target.copy()
                    tauff_target = self.tauff_target.copy()
                
                # 检查关节限制
                for i, (angle, (min_limit, max_limit)) in enumerate(zip(q_target, self.joint_limits)):
                    if angle < min_limit or angle > max_limit:
                        logger_mp.warning(f"关节{i+1}角度{angle:.3f}超出限制[{min_limit:.3f}, {max_limit:.3f}]")
                        q_target[i] = np.clip(angle, min_limit, max_limit)
                
                # 发送控制命令
                if np.any(q_target != self._last_sent_target):
                    self._send_joint_command(q_target)
                    self._last_sent_target = q_target.copy()
                
                # 控制频率
                current_time = time.time()
                elapsed = current_time - start_time
                sleep_time = max(0, self.control_dt - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger_mp.error(f"发布电机命令失败: {e}")
                time.sleep(0.1)
    def get_current_motor_q(self) -> np.ndarray:
        """
        获取当前所有电机关节角度
        """
        return self.get_current_dual_arm_q()
    
    def get_current_motor_dq(self) -> np.ndarray:
        """
        获取当前所有电机关节角速度
        """
        return self.get_current_dual_arm_dq()

    def _get_current_pos(self):
        """
        通过socket获取当前关节角度
        
        Returns:
            np.ndarray: 当前关节角度数组，失败时返回None
        """
        try:
            data = self.robot.recv(1116)
            pos = struct.unpack('!6d', data[252:300])  # 解析32-37位置实际机械臂关节角度
            npPos = np.asarray(pos)
            return npPos
        except Exception as e:
            logger_mp.error(f"获取关节角度失败: {e}")
            return None
    
    def _send_joint_command(self, joints, a=0.1, v=0.1, t=3):
        """
        发送关节角度命令
        
        Args:
            joints: 目标关节角度数组
            a: 加速度
            v: 速度
            t: 时间
        """
        try:
            joints_str = ",".join(map(str, joints))
            strL = "movej([{0}], a={1}, v={2}, t={3}, r=0)\n".format(joints_str, a, v, t).encode()
            logger_mp.debug(f"发送命令: {strL}")
            self.robot.send(strL)
        except Exception as e:
            logger_mp.debug(f"发送关节命令失败: {e}")

    def _go_home_implementation(self):
        """
        实现回到初始位置的具体逻辑
        """
        logger_mp.info("[UR3E] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(self.num_joints)
            self.tauff_target = np.zeros(self.num_joints)
        tolerance = 0.05  # 用于判定关节角度“接近零”的容差阈值，可根据电机精度要求调整
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[UR3E] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)
class AuboI5_ArmController(RobotArmController):
    """
    AuboI5机械臂控制器
    继承自RobotArmController抽象基类
    """
    def __init__(self, robot_ip: str = "192.168.123.113", robot_port: int = 8899, 
                 motion_mode: bool = False):
        """
        初始化AuboI5机械臂控制器
        
        Args:
            robot_ip: 机器人IP地址
            robot_port: 机器人端口
            motion_mode: 是否为运动模式
        """
        super().__init__("AUBO_I5", motion_mode)
        
        # AuboI5特定参数
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.init_joint_angles = np.deg2rad(np.array([0,-36.13,130.63,164.40,82.87,5.0]))  
        # 初始化目标位置和力矩数组（6个关节）
        self.q_target = self.init_joint_angles      # 目标关节角度数组
        self.tauff_target = np.zeros(6)  # 目标关节力矩数组

        # 运动控制参数
        self.arm_velocity_limit = 20.0    # 关节运动速度限制（rad/s）
        self.control_dt = 1.0 / 60.0     # 控制周期（10Hz）
        
        # 渐进加速控制相关参数
        self._speed_gradual_max = False    # 是否启用渐进加速模式
        self._gradual_start_time = None    # 渐进加速开始时间
        self._gradual_time = None          # 渐进加速总时长
        if self.joint_limits is None:
            self.joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        # 初始化机器人连接
        if not self._init_robot_connection():
            raise RuntimeError("无法连接到AuboI5机器人")
            
        # 启动控制线程
        self._start_control_threads()
        
        logger_mp.info("AuboI5机械臂控制器初始化完成")
    
    def _init_robot_connection(self) -> bool:
        """
        初始化AuboI5机器人连接
        """
        try:
            # 系统初始化
            Auboi5Robot.initialize()
            # 创建机械臂控制类
            self.robot = Auboi5Robot()
            # 创建上下文
            handle = self.robot.create_context()
            # 打印上下文
            logger_mp.info("robot.rshd={0}".format(handle))
            result = self.robot.connect(self.robot_ip, self.robot_port)
            time.sleep(1)
            if result != RobotErrorType.RobotError_SUCC:
                print("connect server{0}:{1} failed.".format(self.robot_ip, self.robot_port))
                return False
            logger_mp.info("connect server{0}:{1} success.".format(self.robot_ip, self.robot_port))
            self.robot.project_startup()
            self.robot.enable_robot_event()
            self.robot.init_profile()
            #快速
            joint_maxvelc = (2.596177, 2.596177, 2.596177, 3.110177, 3.110177, 3.110177)
            joint_maxacc = (17.308779/2.5, 17.308779/2.5, 17.308779/2.5, 17.308779/2.5, 17.308779/2.5, 17.308779/2.5)
            #中等速度
            # joint_maxvelc =(0.5192354, 0.5192354, 0.5192354, 0.6220354, 0.6220354, 0.6220354)
            # joint_maxacc = (1.38470232, 1.38470232, 1.38470232, 1.38470232, 1.38470232, 1.38470232)
            #慢速
            # joint_maxacc = (0.5,0.5,0.5,0.5,0.5,0.5)
            # joint_maxvelc = (0.2,0.2,0.2,0.2,0.2,0.2)
            self.robot.set_joint_maxacc(joint_maxacc)
            self.robot.set_joint_maxvelc(joint_maxvelc)
            self.robot.set_arrival_ahead_blend(0.05)
            # 测试连接
            while True:
                test_pos = self._get_current_pos()
                if test_pos is not None:
                    logger_mp.info(f"AuboI5机器人连接成功，当前关节角度: {test_pos}")
                    break
                else:
                    logger_mp.error("无法获取机器人状态，连接失败")
                    time.sleep(0.1)
            return True
        except Exception as e:
            logger_mp.error(f"连接AuboI5机器人失败: {e}")
            return False
    
    def _subscribe_motor_state(self):
        """
        订阅电机状态数据
        """
        while self._subscribe_running:
            try:
                # 获取当前关节状态
                current_pos = self._get_current_pos()
                if current_pos is not None:
                    self.current_joint_angles = current_pos
                    # 计算关节速度（简单差分）
                    if hasattr(self, '_last_positions'):
                        self.current_joint_velocities = (current_pos - self._last_positions) / self.control_dt
                    self._last_positions = current_pos.copy()
                
                # 更新数据缓冲区
                state = LowState(self.num_joints)  # AuboI5有6个关节
                
                # 更新电机状态
                for i, motor_state in enumerate(state.motor_state):
                    motor_state.q = self.current_joint_angles[i] if i < len(self.current_joint_angles) else 0.0
                    motor_state.dq = self.current_joint_velocities[i] if i < len(self.current_joint_velocities) else 0.0
                
                # 保存到数据缓冲区
                self.state_buffer.SetData(state)
                
            except Exception as e:
                logger_mp.error(f"订阅电机状态失败: {e}")
            
            time.sleep(self.control_dt)
    
    def _publish_motor_command(self):
        """
        发布电机控制命令
        """
        if self.motion_mode:
            pass
            
        while self._publish_running:
            try:
                start_time = time.time()
                
                with self.ctrl_lock:
                    q_target = self.q_target.copy()
                    tauff_target = self.tauff_target.copy()
                
                # 检查关节限制
                for i, (angle, (min_limit, max_limit)) in enumerate(zip(q_target, self.joint_limits)):
                    if angle < min_limit or angle > max_limit:
                        logger_mp.warning(f"关节{i+1}角度{angle:.3f}超出限制[{min_limit:.3f}, {max_limit:.3f}]")
                        q_target[i] = np.clip(angle, min_limit, max_limit)
                
                # 应用速度限制
                q_target = self.clip_arm_q_target(q_target, self.arm_velocity_limit)
                
                # 发送控制命令
                if np.any(q_target != self._last_sent_target):
                    self._send_joint_command(q_target)
                    self._last_sent_target = q_target.copy()
                
                # 渐进加速逻辑处理
                if self._speed_gradual_max:
                    t_elapsed = start_time - self._gradual_start_time
                    # 在设定时间内线性增加速度限制
                    self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / self._gradual_time))
                
                # 控制频率
                current_time = time.time()
                elapsed = current_time - start_time
                sleep_time = max(0, self.control_dt - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger_mp.error(f"发布电机命令失败: {e}")
                time.sleep(0.1)
    
    
    def _get_current_pos(self):
        """
        获取当前关节角度
        
        Returns:
            np.ndarray: 当前关节角度数组，失败时返回None
        """
        try:
            current_pos = self.robot.get_current_waypoint()
            return np.array(current_pos['joint'])
        except Exception as e:
            # logger_mp.error(f"获取关节角度失败: {e}")
            return None
    
    def _send_joint_command(self, joints, a=0.1, v=0.1, t=3):
        """
        发送关节角度命令
        
        Args:
            joints: 目标关节角度数组
            a: 加速度
            v: 速度
            t: 时间
        """
        try:
            #self.current_joint_angles 当前关节角度
            #判断角度差小于0.05度则不发送命令
            if np.all(np.abs(self.current_joint_angles - joints) < 0.05):
                logger_mp.debug(f"关节角度差小于0.05度，不发送命令: {joints}")
                return
            joints_radian = (joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
            self.robot.move_joint(joints_radian)
            logger_mp.debug(f"发送AuboI5关节命令: {joints}")
        except Exception as e:
            # logger_mp.error(f"执行失败关节命令: {np.rad2deg(joints)}")
            # logger_mp.error(f"失败末端位姿: {self.current_joint_angles}")
            logger_mp.debug(f"发送关节命令失败: {e}")
    
    def get_current_motor_q(self) -> np.ndarray:
        """
        获取当前所有电机关节角度
        """
        return self.get_current_dual_arm_q()
    
    def get_current_motor_dq(self) -> np.ndarray:
        """
        获取当前所有电机关节角速度
        """
        return self.get_current_dual_arm_dq()
    def _go_home_implementation(self):
        """
        实现回到初始位置的具体逻辑
        """
        logger_mp.info("[AUBO_I5] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = self.init_joint_angles
            self.tauff_target = np.zeros(self.num_joints)
        tolerance = 0.05  # 用于判定关节角度“接近零”的容差阈值，可根据电机精度要求调整
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[AUBO_I5] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)
    
# ============================================================================
# 工厂函数
# ============================================================================

def create_arm_controller(robot_type: str, **kwargs):
    """
    工厂函数：根据机器人类型创建对应的控制器
    
    Args:
        robot_type: 机器人类型 ("UR3E" 或 "FRANKA_PANDA")
        **kwargs: 其他参数
        
    Returns:
        RobotArmController: 对应的机械臂控制器实例
        
    Raises:
        ValueError: 不支持的机器人类型
    """
    robot_type_upper = robot_type.upper()
    
    if robot_type_upper == "UR3E":
        return UR3E_ArmController(**kwargs)
    elif robot_type_upper == "FRANKA_PANDA":
        # 暂时返回UR3E，后续可以添加FrankaPanda实现
        return UR3E_ArmController(**kwargs)
    elif robot_type_upper == "AUBOI5":
        return AuboI5_ArmController(**kwargs)
    else:
        raise ValueError(f"不支持的机器人类型: {robot_type}") 