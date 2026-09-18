# 用于 dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# 用于夹爪（gripper）
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        """
        [注意] *_array 类型的参数需要使用多进程（multiprocessing）Array，因为它需要被传递到内部子进程中

        left_hand_array_in: [输入] 传给 hand_ctrl.control_process 的左手骨架数据（由 XR 设备提供）

        right_hand_array_in: [输入] 传给 hand_ctrl.control_process 的右手骨架数据（由 XR 设备提供）

        dual_hand_data_lock: dual_hand_state_array 和 dual_hand_action_array 的数据同步锁

        dual_hand_state_array_out: [输出] 返回左手(7)、右手(7)的电机状态

        dual_hand_action_array_out: [输出] 返回左手(7)、右手(7)的电机动作

        fps: 控制频率

        Unit_Test: 是否启用单元测试

        simulation_mode: 是否使用仿真模式（默认为 False，即使用真实机器人）
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # 初始化 handcmd 发布者（publisher）和 handstate 订阅者（subscriber）
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # 用于手部状态的共享数组（Shared Arrays）
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # 初始化订阅线程
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # 更新左手状态
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # 更新右手状态
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # id 占 4 位（bit）
            self.status = status & 0x07  # status 占 3 位（bit）
            self.timeout = timeout & 0x01  # timeout 占 1 位（bit）

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """设置当前左、右手电机状态的目标 q（关节角）"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array_in, right_hand_array_in, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # 初始化 dex3-1 的左手命令消息（cmd msg）
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # 初始化 dex3-1 的右手命令消息（cmd msg）
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                # 获取双手状态
                with left_hand_array_in.get_lock():
                    left_hand_data  = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                with right_hand_array_in.get_lock():
                    right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()

                # 从共享数组读取左右 q_state
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # 如果手部数据已完成初始化。
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                # 获取双手动作（action）
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False, simulation_mode = False):
        """
        [注意] *_array 类型的参数需要使用多进程（multiprocessing）Array，因为它需要被传递到内部子进程中

        left_gripper_value_in: [输入] 传给 control_thread 的左夹爪控制数据（由 XR 设备提供）

        right_gripper_value_in: [输入] 传给 control_thread 的右夹爪控制数据（由 XR 设备提供）

        dual_gripper_data_lock: dual_gripper_state_array 和 dual_gripper_action_array 的数据同步锁

        dual_gripper_state_out: [输出] 返回左(1)、右(1)夹爪电机状态

        dual_gripper_action_out: [输出] 返回左(1)、右(1)夹爪电机动作

        fps: 控制频率

        Unit_Test: 是否启用单元测试

        simulation_mode: 是否使用仿真模式（默认为 False，即使用真实机器人）
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode
        
        if filter:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
 
        # 初始化 handcmd 发布者（publisher）和 handstate 订阅者（subscriber）
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # 用于夹爪状态的共享数组（Shared Arrays）
        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        # 初始化订阅线程
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in, self.left_gripper_state_value, self.right_gripper_state_value,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!\n")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
            time.sleep(0.002)
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """设置当前左、右夹爪电机命令（cmd）的目标 q（关节角）"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, left_gripper_state_value, right_gripper_state_value, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True
        if self.simulation_mode:
            DELTA_GRIPPER_CMD = 1.0
        else:   
            DELTA_GRIPPER_CMD = 0.18     # 电机转动 5.4 弧度，夹爪滑台张开 9 cm，因此 0.6 rad <==> 1 cm，0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN  = 0.0           # 启动时夹爪闭合对应的电机最小初始位置。
        RIGHT_MAPPED_MIN = 0.0           # 启动时夹爪闭合对应的电机最小初始位置。
        # 标定前夹爪闭合对应的电机最大初始位置（导轨行程按 0.6 cm/rad * 9 rad = 5.4 cm 计算）。
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # 初始化夹爪命令消息（cmd msg）
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd
        try:
            while self.running:
                start_time = time.time()
                # 从 XR 设备获取双手骨骼点状态
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value

                if left_gripper_value != 0.0 or right_gripper_value != 0.0: # 如果输入数据已完成初始化。
                    # 将 [0, THUMB_INDEX_DISTANCE_MAX] 线性映射到夹爪动作范围
                    left_target_action  = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])

                # 获取当前双夹爪电机状态
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])

                # 对双夹爪动作进行限幅（clip），避免超出范围
                left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)

                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")

class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleop.image_server.image_client import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='选择 XR 设备追踪来源')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='选择末端执行器控制器')
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    # 图像
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # 头部相机分辨率
        'head_camera_id_numbers': [0],
    }
    ASPECT_RATIO_THRESHOLD = 2.0  # 如果宽高比超过该值，则认为是双目
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    # 图像
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)
    img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)
    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television（电视端）：从 XR 设备获取手部位姿数据，并将机器人头部相机图像传输到 XR 设备。
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

# 末端执行器
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [输入]
        right_hand_pos_array = Array('d', 75, lock = True)     # [输入]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [输出] 当前左手、右手 state(14) 数据。
        dual_hand_action_array = Array('d', 14, lock = False)  # [输出] 当前左手、右手 action(14) 数据。
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [输入]
        right_gripper_value = Value('d', 0.0, lock=True)       # [输入]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # 当前左、右夹爪 state(2) 数据。
        dual_gripper_action_array = Array('d', 2, lock=False)  # 当前左、右夹爪 action(2) 数据。
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            tele_data = tv_wrapper.get_motion_state_data()
            if args.ee == "dex3" and args.xr_mode == "hand":
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

            # with dual_hand_data_lock:
            #     logger_mp.info(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
