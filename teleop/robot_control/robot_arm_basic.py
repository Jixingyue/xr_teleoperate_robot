#!/usr/bin/env python3
"""
通用机械臂控制器抽象基类
所有机械臂控制器都应该继承这个基类
"""

import numpy as np
import threading
import time
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import logging_mp

from teleop.utils import config_loader
logger_mp = logging_mp.get_logger(__name__)

# ============================================================================
# 统一数据类定义
# ============================================================================

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class LowState:
    """统一机器人状态类"""
    def __init__(self, num_joints):
        self.motor_state = [MotorState() for _ in range(num_joints)]

class DataBuffer:
    """线程安全数据缓冲区"""
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data
# ============================================================================
# 抽象基类定义
# ============================================================================

class RobotArmController(ABC):
    """
    通用机械臂控制器抽象基类
    
    所有机械臂控制器都应该继承这个基类并实现其抽象方法。
    这个基类定义了机械臂控制的标准接口。
    """
    
    def __init__(self, robot_type: str, motion_mode: bool = False):
        """
        初始化机械臂控制器
        
        Args:
            robot_type: 机器人类型标识符
            motion_mode: 是否为运动模式
        """
        self.robot_type = robot_type
        self.motion_mode = motion_mode
        self.config = config_loader.get_robot_config(robot_type)
        self.joint_arm_index = self.config["arm_joints"]
        self.num_joints = len(self.joint_arm_index)
        # 关节限制
        self.joint_limits = self.config["joint_limits"]
        # 控制参数
        self.control_dt = 1.0 / 250.0  # 控制周期
        self.velocity_limit = 20.0     # 速度限制
        
        # 线程锁
        self.ctrl_lock = threading.Lock()
        
        # 状态标志
        self._initialized = False
        self._running = False
                
        # 线程控制
        self.subscribe_thread = None
        self.publish_thread = None
        self._subscribe_running = False
        self._publish_running = False        

        # 控制参数
        self.kp = 100.0  # 位置增益
        self.kd = 10.0   # 速度增益
        # 数据缓冲区
        self.state_buffer = DataBuffer()
        self.command_buffer = DataBuffer()
        # 当前关节状态
        self.current_joint_angles = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)
        
        # 目标状态
        self.q_target = np.zeros(self.num_joints)
        self.tauff_target = np.zeros(self.num_joints)
        
        # 控制状态
        self._last_sent_target = np.zeros(self.num_joints)
        self._last_positions = np.zeros(self.num_joints)
        logger_mp.info(f"初始化 {robot_type} 机械臂控制器")
    
    @abstractmethod
    def _init_robot_connection(self) -> bool:
        """
        初始化机器人连接
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def _subscribe_motor_state(self):
        """
        订阅电机状态数据
        这个方法应该在子线程中运行
        """
        pass
    
    @abstractmethod
    def _publish_motor_command(self):
        """
        发布电机控制命令
        这个方法应该在子线程中运行
        """
        pass
    
    def get_current_dual_arm_q(self) -> np.ndarray:
        """
        获取当前双臂关节角度（UR3E只有单臂，返回单臂数据）
        """
        return np.array([self.state_buffer.GetData().motor_state[joint["index"]].q for joint in self.joint_arm_index])

    
    def get_current_dual_arm_dq(self) -> np.ndarray:
        """
        获取当前双臂关节角速度（UR3E只有单臂，返回单臂数据）
        """
        return np.array([self.state_buffer.GetData().motor_state[joint["index"]].dq for joint in self.joint_arm_index])
    @abstractmethod
    def get_current_motor_q(self) -> np.ndarray:
        """
        获取当前所有电机关节角度
        """
        pass
    @abstractmethod
    def get_current_motor_dq(self) -> np.ndarray:
        """
        获取当前所有电机关节角速度
        """
        pass
    
    def ctrl_dual_arm(self, q_target, tauff_target):
        '''设置左右臂电机的控制目标值 q 与前馈力矩 tau。'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
    
    
    def ctrl_dual_arm_go_home(self):
        """
        控制双臂回到初始位置
        """
        logger_mp.info(f"[{self.robot_type}] 双臂回到初始位置...")
        self._go_home_implementation()
        logger_mp.info(f"[{self.robot_type}] 双臂已回到初始位置")
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
        while self.state_buffer.GetData() is None:
            time.sleep(0.1)
            print("等待状态数据返回...")
        logger_mp.info(f"[{self.robot_type}] 控制线程已启动")
    
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
        logger_mp.info(f"[{self.robot_type}] 控制线程已停止")

    @abstractmethod
    def _go_home_implementation(self):
        """
        实现回到初始位置的具体逻辑
        """
        pass
    
    def speed_gradual_max(self, t = 5.0):
        '''参数 t 为臂速度逐渐增大到最大值所需的总时间，单位为秒。默认为 5.0。'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''立即将臂速度设置为最大值，而不是逐渐增大。'''
        self.arm_velocity_limit = 30.0
    
    
    def clip_arm_q_target(self, target_q: np.ndarray, velocity_limit: float) -> np.ndarray:
        """
        限制关节角度目标值以避免过大的速度
        
        Args:
            target_q: 目标关节角度
            velocity_limit: 速度限制
            
        Returns:
            np.ndarray: 限制后的目标关节角度
        """
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        clipped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return clipped_arm_q_target
    
    def start(self):
        """
        启动控制器
        """
        if not self._initialized:
            if self._init_robot_connection():
                self._initialized = True
                self._running = True
                self._start_control_threads()
                logger_mp.info(f"[{self.robot_type}] 控制器启动成功")
            else:
                logger_mp.error(f"[{self.robot_type}] 控制器启动失败")
        else:
            logger_mp.warning(f"[{self.robot_type}] 控制器已经启动")
    
    def stop(self):
        """
        停止控制器
        """
        if self._running:
            self.ctrl_dual_arm_go_home()
            self._running = False
            self._stop_control_threads()
            logger_mp.info(f"[{self.robot_type}] 控制器已停止")
        else:
            logger_mp.warning(f"[{self.robot_type}] 控制器已经停止")
    
    def is_connected(self) -> bool:
        """
        检查是否已连接
        
        Returns:
            bool: 是否已连接
        """
        return self._initialized and self._running
    
    def get_robot_info(self) -> Dict[str, Any]:
        """
        获取机器人信息
        
        Returns:
            Dict[str, Any]: 机器人信息字典
        """
        return {
            'robot_type': self.robot_type,
            'motion_mode': self.motion_mode,
            'connected': self.is_connected(),
            'control_dt': self.control_dt,
            'velocity_limit': self.velocity_limit
        }
    
    def _is_moving(self) -> bool:
        """
        检查机器人是否在运动
        
        Returns:
            bool: 是否在运动
        """
        # 子类可以重写此方法来实现具体的运动检测逻辑
        return False