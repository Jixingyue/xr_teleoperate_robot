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
from robot_control.robot_arm_basic import (
    RobotArmController, MotorState, LowState, DataBuffer
)

import logging_mp

from teleop.utils import config_loader
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
        
        # 控制参数
        self.kp = 100.0  # 位置增益
        self.kd = 10.0   # 速度增益
        
        # 当前关节状态
        self.current_joint_angles = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)
        
        # 目标状态
        self.q_target = np.zeros(self.num_joints)
        self.tauff_target = np.zeros(self.num_joints)
        
        # 线程控制
        self.subscribe_thread = None
        self.publish_thread = None
        self._subscribe_running = False
        self._publish_running = False
        
        # 数据缓冲区
        self.data_buffer = DataBuffer()
        self.data_lock = threading.Lock()
        
        # 控制状态
        self._last_sent_target = np.zeros(self.num_joints)
        self._last_positions = np.zeros(self.num_joints)
        
    def _init_robot_connection(self) -> bool:
        """
        初始化UR3E机器人连接
        """
        try:
            logger_mp.info(f"连接到UR3E机器人: {self.robot_ip}:{self.robot_port}")
            robot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            robot.connect((self.robot_ip, self.robot_port))
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
            try:
                # 通过socket获取真实关节状态
                current_pos = self._get_current_pos()
                if current_pos is not None:
                    self.current_joint_angles = current_pos
                    # 计算关节速度（简单差分）
                    if hasattr(self, '_last_positions'):
                        dt = 0.002  # 500Hz更新频率
                        self.current_joint_velocities = (current_pos - self._last_positions) / dt
                    self._last_positions = current_pos.copy()
                
                # 更新数据缓冲区
                with self.data_lock:
                    # 创建统一状态对象
                    state = LowState(6)  # UR3E有6个关节
                    
                    # 更新电机状态
                    for i, motor_state in enumerate(state.motor_state):
                        motor_state.q = self.current_joint_angles[i] if i < len(self.current_joint_angles) else 0.0
                        motor_state.dq = self.current_joint_velocities[i] if i < len(self.current_joint_velocities) else 0.0
                    
                    # 保存到数据缓冲区
                    self.data_buffer.SetData(state)
                
                time.sleep(0.002)  # 500Hz更新频率
                
            except Exception as e:
                logger_mp.error(f"订阅电机状态失败: {e}")
                time.sleep(0.1)
    
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
    def _start_control_threads(self):
        """
        启动控制线程
        """
        self._subscribe_running = True
        self._publish_running = True
        
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state, daemon=True)
        self.publish_thread = threading.Thread(target=self._publish_motor_command, daemon=True)
        
        self.subscribe_thread.start()
        self.publish_thread.start()
        
        logger_mp.info("[UR3E] 控制线程已启动")
    
    def _stop_control_threads(self):
        """
        停止控制线程
        """
        self._subscribe_running = False
        self._publish_running = False
        if self.subscribe_thread and self.subscribe_thread.is_alive():
            self.subscribe_thread.join(timeout=1.0)
        if self.publish_thread and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)
        self.robot.close()
        logger_mp.info("[UR3E] 控制线程已停止")

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
    
    def _send_joint_command(self, joints, a=0.9, v=0.9, t=1):
        """
        发送关节角度命令
        
        Args:
            joints: 目标关节角度数组
            a: 加速度
            v: 速度
            t: 时间
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)  # 设置超时时间
            s.connect((self.robot_ip, self.robot_port))  # 使用命令端口
            joints_str = ",".join(map(str, joints))
            strL = "movej([{0}], a={1}, v={2}, t={3}, r=0)\n".format(joints_str, a, v, t).encode()
            logger_mp.debug(f"发送命令: {strL}")
            s.send(strL)
            s.close()
        except Exception as e:
            logger_mp.error(f"发送关节命令失败: {e}")

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
        tolerance = 0.05  # 用于判断关节角是否“接近零”的容差阈值，可根据电机的精度要求调整
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[UR3E] both arms have reached the home position.")
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
    else:
        raise ValueError(f"不支持的机器人类型: {robot_type}") 