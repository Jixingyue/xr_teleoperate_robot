# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
简易仿真状态订阅者类
订阅 rt/sim_state_cmd topic/话题并写入 shared memory/共享内存
"""

import threading
import time
import json
from multiprocessing import shared_memory
from typing import Any, Dict, Optional
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class SharedMemoryManager:
    """shared memory/共享内存管理器"""
    
    def __init__(self, name: Optional[str] = None, size: int = 512):
        """初始化 shared memory/共享内存管理器
        
        Args:
            name: shared memory/共享内存名称，如果为 None 则创建新的共享内存
            size: shared memory/共享内存大小（字节）
        """
        self.size = size
        self.lock = threading.RLock()  # 可重入锁
        
        if name:
            try:
                self.shm = shared_memory.SharedMemory(name=name)
                self.shm_name = name
                self.created = False
            except FileNotFoundError:
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                self.shm_name = self.shm.name
                self.created = True
        else:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            self.shm_name = self.shm.name
            self.created = True
    
    def write_data(self, data: Dict[str, Any]) -> bool:
        """向 shared memory/共享内存写入数据
        
        Args:
            data: 要写入的数据
            
        Returns:
            bool: 是否写入成功
        """
        try:
            with self.lock:
                json_str = json.dumps(data)
                json_bytes = json_str.encode('utf-8')
                
                if len(json_bytes) > self.size - 8:  # 预留 8 字节用于存放长度和时间戳
                    logger_mp.warning(f"Data too large for shared memory ({len(json_bytes)} > {self.size - 8})")
                    return False
                
                # 写入时间戳（4 字节）和数据长度（4 字节）
                timestamp = int(time.time()) & 0xFFFFFFFF  # 32 位时间戳，使用位掩码确保数值在范围内
                self.shm.buf[0:4] = timestamp.to_bytes(4, 'little')
                self.shm.buf[4:8] = len(json_bytes).to_bytes(4, 'little')
                
                # 写入数据
                self.shm.buf[8:8+len(json_bytes)] = json_bytes
                return True
                
        except Exception as e:
            logger_mp.error(f"Error writing to shared memory: {e}")
            return False
    
    def read_data(self) -> Optional[Dict[str, Any]]:
        """从 shared memory/共享内存读取数据
        
        Returns:
            Dict[str, Any]: 读取到的数据字典，失败时返回 None
        """
        try:
            with self.lock:
                # 读取时间戳和数据长度
                timestamp = int.from_bytes(self.shm.buf[0:4], 'little')
                data_len = int.from_bytes(self.shm.buf[4:8], 'little')
                
                if data_len == 0:
                    return None
                
                # 读取数据
                json_bytes = bytes(self.shm.buf[8:8+data_len])
                data = json.loads(json_bytes.decode('utf-8'))
                data['_timestamp'] = timestamp  # 补充时间戳信息
                return data
                
        except Exception as e:
            logger_mp.error(f"Error reading from shared memory: {e}")
            return None
    
    def get_name(self) -> str:
        """获取 shared memory/共享内存名称"""
        return self.shm_name
    
    def cleanup(self):
        """清理 shared memory/共享内存"""
        if hasattr(self, 'shm') and self.shm:
            self.shm.close()
            if self.created:
                try:
                    self.shm.unlink()
                except:
                    pass
    
    def __del__(self):
        """析构函数"""
        self.cleanup()

class SimStateSubscriber:
    """简易仿真状态订阅者类"""
    
    def __init__(self, shm_name: str = "sim_state_cmd_data", shm_size: int = 3096):
        """初始化订阅者
        
        Args:
            shm_name: shared memory/共享内存名称
            shm_size: shared memory/共享内存大小
        """
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.running = False
        self.subscriber = None
        self.subscribe_thread = None
        self.shared_memory = None
        
        # 初始化 shared memory/共享内存
        self._setup_shared_memory()
        
        logger_mp.debug(f"[SimStateSubscriber] Initialized with shared memory: {shm_name}")
    
    def _setup_shared_memory(self):
        """配置 shared memory/共享内存"""
        try:
            self.shared_memory = SharedMemoryManager(self.shm_name, self.shm_size)
            logger_mp.debug(f"[SimStateSubscriber] Shared memory setup successfully")
        except Exception as e:
            logger_mp.error(f"[SimStateSubscriber] Failed to setup shared memory: {e}")
    
    def start_subscribe(self):
        """开始订阅"""
        if self.running:
            logger_mp.warning(f"[SimStateSubscriber] Already running")
            return
        
        try:
            self.subscriber = ChannelSubscriber("rt/sim_state", String_)
            self.subscriber.Init()
            self.running = True

            self.subscribe_thread = threading.Thread(target=self._subscribe_sim_state, daemon=True)
            self.subscribe_thread.start()

            logger_mp.info(f"[SimStateSubscriber] Started subscribing to rt/sim_state")
            
        except Exception as e:
            logger_mp.error(f"[SimStateSubscriber] Failed to start subscribing: {e}")
            self.running = False

    def _subscribe_sim_state(self):
        """订阅循环线程"""
        logger_mp.debug(f"[SimStateSubscriber] Subscribe thread started")
        
        while self.running:
            try:
                if self.subscriber:
                    msg = self.subscriber.Read()
                    if msg:
                        data = json.loads(msg.data)
                    else:
                        logger_mp.warning("[SimStateSubscriber] Received None message")
                    if self.shared_memory and data:
                        self.shared_memory.write_data(data)
                else:
                    logger_mp.error("[SimStateSubscriber] Subscriber is not initialized")
                time.sleep(0.002)
            except Exception as e:
                logger_mp.error(f"[SimStateSubscriber] Error in subscribe loop: {e}")
                time.sleep(0.01)

    def stop_subscribe(self):
        """停止订阅"""
        if not self.running:
            logger_mp.warning(f"[SimStateSubscriber] Already stopped or not running")
            return

        self.running = False
        # 等待线程结束
        if self.subscribe_thread:
            self.subscribe_thread.join(timeout=1.0)

        if self.shared_memory:
            self.shared_memory.cleanup()
        logger_mp.info(f"[SimStateSubscriber] Subscriber stopped")
    
    def read_data(self) -> Optional[Dict[str, Any]]:
        """从 shared memory/共享内存读取数据
        
        Returns:
            Dict: 接收到的数据，无数据或出错时为 None
        """
        try:
            if self.shared_memory:
                return self.shared_memory.read_data()
            return None
        except Exception as e:
            logger_mp.error(f"[SimStateSubscriber] Error reading data: {e}")
            return None
    
    def is_running(self) -> bool:
        """检查订阅者是否正在运行"""
        return self.running
    
    def __del__(self):
        """析构函数"""
        self.stop_subscribe()


def start_sim_state_subscribe(shm_name: str = "sim_state_cmd_data", shm_size: int = 3096) -> SimStateSubscriber:
    """开始仿真状态订阅
    
    Args:
        shm_name: shared memory/共享内存名称  
        shm_size: shared memory/共享内存大小
        
    Returns:
        SimStateSubscriber: 已启动的订阅者实例
    """
    subscriber = SimStateSubscriber(shm_name, shm_size)
    subscriber.start_subscribe()
    return subscriber


# if __name__ == "__main__":
#     # 示例用法
#     logger_mp.info("Starting sim state subscriber...")
#     ChannelFactoryInitialize(0)
#     # 创建并启动订阅者
#     subscriber = start_sim_state_subscribe()
    
#     try:
#         # 持续运行并检查数据
#         while True:
#             data = subscriber.read_data()
#             if data:
#                 logger_mp.info(f"Read data: {data}")
#             time.sleep(1)
            
#     except KeyboardInterrupt:
#         logger_mp.warning("\nInterrupted by user")
#     finally:
#         subscriber.stop_subscribe()
#         logger_mp.info("Subscriber stopped") 