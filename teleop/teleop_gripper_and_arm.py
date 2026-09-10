#!/usr/bin/env python3
"""
真机单机械臂遥操作脚本
专门针对单机械臂和夹爪控制
"""

import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from robot_control.universal_robot_arm_ik import UniversalRobotArmIK as ArmIK
from robot_control.universal_robot_arm import create_arm_controller
from robot_control.robot_gripper_inspire import Inspire_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from sshkeyboard import listen_keyboard, stop_listening

# 状态转换
start_signal = False
running = True
should_toggle_recording = False
is_recording = False

def on_press(key):
    global running, start_signal, should_toggle_recording
    if key == 'r':
        start_signal = True
        logger_mp.info("程序启动信号已接收。")
    elif key == 'q':
        stop_listening()
        running = False
    elif key == 's':
        should_toggle_recording = True
    else:
        logger_mp.info(f"按键 {key} 被按下，但未定义此按键的操作。")

listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
listen_keyboard_thread.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UR3E单机械臂真机遥操作')
    parser.add_argument('--task_dir', type=str, default='./utils/data', help='数据保存路径')
    parser.add_argument('--frequency', type=float, default=60.0, help='数据保存频率')
    
    # 基本控制参数
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='controller', help='选择XR设备追踪源')
    parser.add_argument('--arm', type=str, choices=['UR3E', 'FRANKA_PANDA',"AUBOI5"], default='AUBOI5', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['inspire',"inspire_gripper"], default='inspire_gripper', help='Select end effector controller')
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    # 模式标志
    parser.add_argument('--record', action='store_true', help='启用数据记录')
    parser.add_argument('--headless', action='store_true', help='启用无头模式（无显示）')
    
    args = parser.parse_args()
    logger_mp.info(f"参数: {args}")
    
    # 图像配置
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 640],  # 头部相机分辨率
        'head_camera_id_numbers': [7],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # 手腕相机分辨率
        'wrist_camera_id_numbers': [4],
    }
    
    # 计算图像形状
    ASPECT_RATIO_THRESHOLD = 2.0
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
    
    # 创建共享内存
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
    
    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name, 
                                 wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)
    
    # 启动图像接收线程
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()
    
    # 初始化Television包装器
    tv_wrapper = TeleVuerWrapper(
        binocular=BINOCULAR, 
        use_hand_tracking=args.xr_mode == "hand", 
        img_shape=tv_img_shape, 
        img_shm_name=tv_img_shm.name, 
        return_state_data=True, 
        return_hand_rot_data=False
    )
    
    # 初始化机械臂控制器
    if args.arm == "UR3E":
        arm_ctrl = create_arm_controller(robot_type=args.arm, motion_mode=args.motion,robot_ip="192.168.123.17")
    elif args.arm == "FRANKA_PANDA":
        arm_ctrl = create_arm_controller(robot_type=args.arm, motion_mode=args.motion)
    elif args.arm == "AUBOI5":
        arm_ctrl = create_arm_controller(robot_type=args.arm, motion_mode=args.motion)
    else:
        raise ValueError(f"Unsupported arm type: {args.arm}")
    arm_ik = ArmIK(robot_type=args.arm)
    # end-effector
    if args.ee == "inspire":
        left_gripper_value = Value('d', 1.0, lock = True)      # [input]
        right_gripper_value = Value('d', 1.0, lock = True)     # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock = False)   # [output] current left, right gripper state data.
        dual_gripper_action_array = Array('d', 2, lock = False)  # [output] current left, right gripper action data.
        gripper_ctrl = Inspire_Controller(None, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
    elif args.ee == "inspire_gripper":
        left_gripper_value = Value('d', 1.0, lock = True)      # [input]
        right_gripper_value = Value('d', 1.0, lock = True)     # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock = False)   # [output] current left, right gripper state data.
        dual_gripper_action_array = Array('d', 2, lock = False)  # [output] current left, right gripper action data.
        gripper_ctrl = Inspire_Gripper_Controller(None, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
    else:
        pass
    # 记录模式
    if args.record and args.headless:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=False)
    elif args.record and not args.headless:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=True)
    
    try:
        logger_mp.info("请输入启动信号（按'r'开始后续程序）")
        while not start_signal:
            time.sleep(0.01)
        
        # 启动机械臂控制器
        arm_ctrl.start()
        
        # 机械臂回到初始位置
        arm_ctrl.ctrl_dual_arm_go_home()
        
        while running:
            start_time = time.time()
            
            # 显示图像（非无头模式）
            if not args.headless:
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("遥操作图像", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s'):
                    should_toggle_recording = True
                elif key == ord('h'):
                    arm_ctrl.ctrl_dual_arm_go_home()
            
            # 记录控制
            if args.record and should_toggle_recording:
                should_toggle_recording = False
                if not is_recording:
                    if recorder.create_episode():
                        is_recording = True
                        logger_mp.info("开始记录数据")
                    else:
                        logger_mp.error("创建记录失败，未开始记录")
                else:
                    is_recording = False
                    recorder.save_episode()
                    logger_mp.info("停止记录数据")
            
            # 获取遥操作数据
            tele_data = tv_wrapper.get_motion_state_data()
            
            # 处理夹爪控制
            if args.xr_mode == "controller":
                with dual_gripper_data_lock:
                    left_gripper_value.value = tele_data.left_trigger_value
                    right_gripper_value.value = tele_data.right_trigger_value

            elif args.xr_mode == "hand":
                with dual_gripper_data_lock:
                    left_gripper_value.value = tele_data.left_pinch_value
                    right_gripper_value.value = tele_data.right_pinch_value
            
            # 获取当前机械臂状态
            current_arm_q = arm_ctrl.get_current_dual_arm_q()
            current_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            
            # 求解逆运动学
            time_ik_start = time.time()
            sol_q, sol_tauff = arm_ik.solve_ik(
                right_wrist=tele_data.right_arm_pose, 
                current_lr_arm_motor_q=current_arm_q, 
                current_lr_arm_motor_dq=current_arm_dq
            )
            time_ik_end = time.time()
            logger_mp.debug(f"逆运动学求解时间: {round(time_ik_end - time_ik_start, 6)}秒")
            
            # 控制机械臂
            if sol_q is not None:
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            else:
                logger_mp.warning("逆运动学求解失败")
            
            # 记录数据
            if args.record and is_recording:
                # 获取当前图像
                current_tv_image = tv_img_array.copy()
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                
                # 准备状态和动作数据
                left_arm_state = current_arm_q[:6]
                right_arm_state = current_arm_q[6:]
                left_arm_action = sol_q[:6]
                right_arm_action = sol_q[6:]
                left_gripper_state = dual_gripper_state_array[0]
                right_gripper_state = dual_gripper_state_array[1]
                left_gripper_action= dual_gripper_action_array[0]
                right_gripper_action = dual_gripper_action_array[1]
                
                # 准备图像数据
                colors = {}
                depths = {}
                if BINOCULAR:
                    colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                    colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                    if WRIST:
                        colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                        colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                else:
                    colors[f"color_{0}"] = current_tv_image
                    if WRIST:
                        colors[f"color_{1}"] = current_wrist_image                
                # 状态数据
                states = {
                    "right_arm": {
                        "qpos": right_arm_state.tolist(),
                        "qvel": [],
                        "torque": [],
                    },
                    "right_ee": {
                        "qpos": [right_gripper_value.value],
                        "qvel": [],
                        "torque": [],
                    },
                    "body": {
                        "qpos": [],
                    },
                }
                
                # 动作数据
                actions = {
                    "right_arm": {
                        "qpos": right_arm_action.tolist(),
                        "qvel": [],
                        "torque": [],
                    },
                    "right_ee": {
                        "qpos": [right_gripper_value.value],
                        "qvel": [],
                        "torque": [],
                    },
                    "body": {
                        "qpos": [],
                    },
                }
                
                recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)
            
            # 控制循环频率
            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"主循环睡眠时间: {sleep_time}")
    
    except KeyboardInterrupt:
        logger_mp.info("键盘中断，正在退出程序...")
    finally:
        # 清理资源
        arm_ctrl.stop()
        
        tv_img_shm.close()
        tv_img_shm.unlink()
        
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        
        if args.record:
            recorder.close()
        
        listen_keyboard_thread.join()
        logger_mp.info("程序已退出")
        exit(0) 