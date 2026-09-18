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
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from sshkeyboard import listen_keyboard, stop_listening

# 用于仿真
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int,publisher): # 场景重置信号
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# 状态转换
start_signal = False
running = True
should_toggle_recording = False
is_recording = False
def on_press(key):
    global running, start_signal, should_toggle_recording
    if key == 'r':
        start_signal = True
        logger_mp.info("Program start signal received.")
    elif key == 'q':
        stop_listening()
        running = False
    elif key == 's':
        should_toggle_recording = True
    else:
        logger_mp.info(f"{key} was pressed, but no action is defined for this key.")
listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
listen_keyboard_thread.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = '数据保存路径')
    parser.add_argument('--frequency', type = float, default = 60.0, help = '数据保存频率')

    # 基本控制参数
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='controller', help='选择 XR 设备追踪源')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='选择机械臂控制器')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='选择末端执行器控制器')
    # 模式标志
    parser.add_argument('--record', action = 'store_true', help = '启用数据记录')
    parser.add_argument('--motion', action = 'store_true', help = '启用运动控制模式')
    parser.add_argument('--headless', action='store_true', help='启用无头模式（无显示）')
    parser.add_argument('--sim', action = 'store_true', help = '启用 Isaac 仿真模式')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # 图像客户端：img_config 应与 image_server.py（机器人开发计算单元上）中的配置保持一致
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 640],  # 头部相机分辨率
            'head_camera_id_numbers': [0],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],  # 手腕相机分辨率
            'wrist_camera_id_numbers': [2, 4],
        }
    else:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 1280],  # 头部相机分辨率
            'head_camera_id_numbers': [0],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],  # 手腕相机分辨率
            'wrist_camera_id_numbers': [2, 4],
        }


    ASPECT_RATIO_THRESHOLD = 2.0 # 若宽高比超过该值，则视为双目
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

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST and args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address="127.0.0.1")
    elif WRIST and not args.sim:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television：从 XR 设备获取手部位姿数据，并将机器人头部相机图像传输到 XR 设备。
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

    # 机械臂
    if args.arm == "G1_29":
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        arm_ik = G1_29_ArmIK()
    elif args.arm == "G1_23":
        arm_ctrl = G1_23_ArmController(simulation_mode=args.sim)
        arm_ik = G1_23_ArmIK()
    elif args.arm == "H1_2":
        arm_ctrl = H1_2_ArmController(simulation_mode=args.sim)
        arm_ik = H1_2_ArmIK()
    elif args.arm == "H1":
        arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        arm_ik = H1_ArmIK()

    # 末端执行器
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [输入]
        right_hand_pos_array = Array('d', 75, lock = True)     # [输入]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [输出] 当前左、右手状态（14）数据。
        dual_hand_action_array = Array('d', 14, lock = False)  # [输出] 当前左、右手动作（14）数据。
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [输入]
        right_gripper_value = Value('d', 0.0, lock=True)       # [输入]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # 当前左、右夹爪状态（2）数据。
        dual_gripper_action_array = Array('d', 2, lock=False)  # 当前左、右夹爪动作（2）数据。
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
    elif args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock = True)      # [输入]
        right_hand_pos_array = Array('d', 75, lock = True)     # [输入]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [输出] 当前左、右手状态（12）数据。
        dual_hand_action_array = Array('d', 12, lock = False)  # [输出] 当前左、右手动作（12）数据。
        hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "brainco":
        left_hand_pos_array = Array('d', 75, lock = True)      # [输入]
        right_hand_pos_array = Array('d', 75, lock = True)     # [输入]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [输出] 当前左、右手状态（12）数据。
        dual_hand_action_array = Array('d', 12, lock = False)  # [输出] 当前左、右手动作（12）数据。
        hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    else:
        pass

    # 仿真模式
    if args.sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from teleop.utils.sim_state_topic import start_sim_state_subscribe
        sim_state_subscriber = start_sim_state_subscribe()

    # 控制器 + 运动模式
    if args.xr_mode == "controller" and args.motion:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        sport_client = LocoClient()
        sport_client.SetTimeout(0.0001)
        sport_client.Init()
    
    # 记录 + 无头模式
    if args.record and args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = False)
    elif args.record and not args.headless:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        
    try:
        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        while not start_signal:
            time.sleep(0.01)
        arm_ctrl.speed_gradual_max()
        while running:
            start_time = time.time()

            if not args.headless:
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)
                elif key == ord('s'):
                    should_toggle_recording = True
                elif key == ord('a'):
                    if args.sim:
                        publish_reset_category(2, reset_pose_publisher)

            if args.record and should_toggle_recording:
                should_toggle_recording = False
                if not is_recording:
                    if recorder.create_episode():
                        is_recording = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    is_recording = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            # 获取输入数据
            tele_data = tv_wrapper.get_motion_state_data()
            if (args.ee == "dex3" or args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_trigger_value
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_trigger_value
            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_pinch_value
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_pinch_value
            else:
                pass        
            
            # 高层控制
            if args.xr_mode == "controller" and args.motion:
                # 退出遥操作
                if tele_data.tele_state.right_aButton:
                    running = False
                    stop_listening()
                # 命令机器人进入阻尼模式，即软急停功能
                if tele_data.tele_state.left_thumbstick_state and tele_data.tele_state.right_thumbstick_state:
                    sport_client.Damp()
                # 控制，将速度限制在 0.3 以内
                sport_client.Move(-tele_data.tele_state.left_thumbstick_value[1]  * 0.3,
                                  -tele_data.tele_state.left_thumbstick_value[0]  * 0.3,
                                  -tele_data.tele_state.right_thumbstick_value[0] * 0.3)

            # 获取当前机器人状态数据。
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # 利用电机数据和手腕位姿求解 IK/逆运动学，再用 IK 结果控制机械臂。
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # 记录数据
            if args.record:
                # 灵巧手或夹爪
                if args.ee == "dex3" and args.xr_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.xr_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.xr_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.tele_state.left_thumbstick_value[1]  * 0.3,
                                               -tele_data.tele_state.left_thumbstick_value[0]  * 0.3,
                                               -tele_data.tele_state.right_thumbstick_value[0] * 0.3]
                elif (args.ee == "inspire1" or args.ee == "brainco") and args.xr_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []
                # 头部图像
                current_tv_image = tv_img_array.copy()
                # 手腕图像
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                # 机械臂状态与动作
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if is_recording:
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
                            colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array 转为 list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        if args.sim:
            sim_state_subscriber.stop_subscribe()
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        if args.record:
            recorder.close()
        listen_keyboard_thread.join()
        logger_mp.info("Finally, exiting program...")
        exit(0)
