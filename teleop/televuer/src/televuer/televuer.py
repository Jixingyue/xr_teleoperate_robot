from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import cv2
import os
from pathlib import Path


class TeleVuer:
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape, img_shm_name, cert_file=None, key_file=None, ngrok=False, webrtc=False):
        """
        TeleVuer 类，用于基于 OpenXR 的 XR 遥操作应用。
        本类负责与 Vuer 服务器通信，并管理用于图像和位姿数据的共享内存。

        :param binocular: bool，应用为双目（立体视觉）还是单目。
        :param use_hand_tracking: bool，使用手部追踪还是手柄追踪。
        :param img_shape: tuple，图像形状（高度、宽度、通道数）。
        :param img_shm_name: str，图像共享内存的名称。
        :param cert_file: str，SSL 证书文件路径。
        :param key_file: str，SSL 密钥文件路径。
        :param ngrok: bool，是否使用 ngrok 进行内网穿透。
        """
        self.binocular = binocular
        self.use_hand_tracking = use_hand_tracking
        self.img_height = img_shape[0]
        if self.binocular:
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]
        
        current_module_dir = Path(__file__).resolve().parent.parent.parent
        if cert_file is None:
            cert_file = os.path.join(current_module_dir, "cert.pem")
        if key_file is None:
            key_file = os.path.join(current_module_dir, "key.pem")

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)

        self.webrtc = webrtc
        if self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        elif not self.binocular and not self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_monocular)
        elif self.webrtc:
            self.vuer.spawn(start=False)(self.main_image_webrtc)

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_pinch_state_shared = Value('b', False, lock=True)
            self.left_pinch_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)

            self.right_pinch_state_shared = Value('b', False, lock=True)
            self.right_pinch_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
        else:
            self.left_trigger_state_shared = Value('b', False, lock=True)
            self.left_trigger_value_shared = Value('d', 0.0, lock=True)
            self.left_squeeze_state_shared = Value('b', False, lock=True)
            self.left_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.left_thumbstick_state_shared = Value('b', False, lock=True)
            self.left_thumbstick_value_shared = Array('d', 2, lock=True)
            self.left_aButton_shared = Value('b', False, lock=True)
            self.left_bButton_shared = Value('b', False, lock=True)

            self.right_trigger_state_shared = Value('b', False, lock=True)
            self.right_trigger_value_shared = Value('d', 0.0, lock=True)
            self.right_squeeze_state_shared = Value('b', False, lock=True)
            self.right_squeeze_value_shared = Value('d', 0.0, lock=True)
            self.right_thumbstick_state_shared = Value('b', False, lock=True)
            self.right_thumbstick_value_shared = Array('d', 2, lock=True)
            self.right_aButton_shared = Value('b', False, lock=True)
            self.right_bButton_shared = Value('b', False, lock=True)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    def vuer_run(self):
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        try:
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]

            left_controller_state = event.value["leftState"]
            right_controller_state = event.value["rightState"]

            def extract_controller_states(state_dict, prefix):
                # 扳机
                with getattr(self, f"{prefix}_trigger_state_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_state_shared").value = bool(state_dict.get("trigger", False))
                with getattr(self, f"{prefix}_trigger_value_shared").get_lock():
                    getattr(self, f"{prefix}_trigger_value_shared").value = float(state_dict.get("triggerValue", 0.0))
                # 握紧
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))
                # 摇杆
                with getattr(self, f"{prefix}_thumbstick_state_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_state_shared").value = bool(state_dict.get("thumbstick", False))
                with getattr(self, f"{prefix}_thumbstick_value_shared").get_lock():
                    getattr(self, f"{prefix}_thumbstick_value_shared")[:] = state_dict.get("thumbstickValue", [0.0, 0.0])
                # 按键
                with getattr(self, f"{prefix}_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_aButton_shared").value = bool(state_dict.get("aButton", False))
                with getattr(self, f"{prefix}_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_bButton_shared").value = bool(state_dict.get("bButton", False))

            extract_controller_states(left_controller_state, "left")
            extract_controller_states(right_controller_state, "right")
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand_state = event.value["leftState"]
            right_hand_state = event.value["rightState"]

            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hand_states(state_dict, prefix):
                # 捏合
                with getattr(self, f"{prefix}_pinch_state_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_state_shared").value = bool(state_dict.get("pinch", False))
                with getattr(self, f"{prefix}_pinch_value_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_value_shared").value = float(state_dict.get("pinchValue", 0.0))
                # 握紧
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hand_states(left_hand_state, "left")
            extract_hand_states(right_hand_state, "right")

        except:
            pass
    
    async def main_image_binocular(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image[:, :self.img_width],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        # 底层渲染引擎为物体和相机都支持图层二进制位掩码（layer binary bitmask）。
                        # 下面我们将左右两个图像平面分别设置为 layers=1 和 layers=2。
                        # 注意，这两个掩码分别关联左眼相机和右眼相机。
                        layers=1,
                        format="jpeg",
                        quality=100,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        display_image[:, self.img_width:],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=100,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 采用 'jpeg' 编码，配合两次发送之间 16ms 的等待，帧率约为 30fps。
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    showLeft=True,
                    showRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    async def main_image_webrtc(self, session, fps=60):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    showLeft=True,
                    showRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    showLeft=False,
                    showRight=False,
                )
            )
    
        session.upsert(
            WebRTCVideoPlane(
            # WebRTCStereoVideoPlane(
                src="https://10.0.7.49:8080/offer",
                iceServer={},
                key="webrtc",
                aspect=1.778,
                height = 7,
            ),
            to="bgChildren",
        )
        while True:
            await asyncio.sleep(1)
    # ==================== 通用数据 ====================
    @property
    def head_pose(self):
        """np.ndarray，shape (4, 4)，来自 Vuer 的头部 SE(3) 位姿矩阵（基：OpenXR 约定）。"""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray，shape (4, 4)，来自 Vuer 的左臂 SE(3) 位姿矩阵（基：OpenXR 约定）。"""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray，shape (4, 4)，来自 Vuer 的右臂 SE(3) 位姿矩阵（基：OpenXR 约定）。"""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== 手部追踪数据 ====================
    @property
    def left_hand_positions(self):
        """np.ndarray，shape (25, 3)，左手 25 个关键点的三维位置。"""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray，shape (25, 3)，右手 25 个关键点的三维位置。"""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray，shape (25, 3, 3)，左手 25 个关键点的朝向（展平的 3x3 矩阵，列优先）。"""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray，shape (25, 3, 3)，右手 25 个关键点的朝向（展平的 3x3 矩阵，列优先）。"""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch_state(self):
        """bool，左手是否处于捏合状态。"""
        with self.left_pinch_state_shared.get_lock():
            return self.left_pinch_state_shared.value

    @property
    def left_hand_pinch_value(self):
        """float，左手的捏合力度。"""
        with self.left_pinch_value_shared.get_lock():
            return self.left_pinch_value_shared.value

    @property
    def left_hand_squeeze_state(self):
        """bool，左手是否处于握紧状态。"""
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_hand_squeeze_value(self):
        """float，左手的握紧力度。"""
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def right_hand_pinch_state(self):
        """bool，右手是否处于捏合状态。"""
        with self.right_pinch_state_shared.get_lock():
            return self.right_pinch_state_shared.value

    @property
    def right_hand_pinch_value(self):
        """float，右手的捏合力度。"""
        with self.right_pinch_value_shared.get_lock():
            return self.right_pinch_value_shared.value

    @property
    def right_hand_squeeze_state(self):
        """bool，右手是否处于握紧状态。"""
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_hand_squeeze_value(self):
        """float，右手的握紧力度。"""
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    # ==================== 手柄数据 ====================
    @property
    def left_controller_trigger_state(self):
        """bool，左手柄扳机是否按下。"""
        with self.left_trigger_state_shared.get_lock():
            return self.left_trigger_state_shared.value

    @property
    def left_controller_trigger_value(self):
        """float，左手柄扳机的模拟量数值（0.0 ~ 1.0）。"""
        with self.left_trigger_value_shared.get_lock():
            return self.left_trigger_value_shared.value

    @property
    def left_controller_squeeze_state(self):
        """bool，左手柄握紧键是否按下。"""
        with self.left_squeeze_state_shared.get_lock():
            return self.left_squeeze_state_shared.value

    @property
    def left_controller_squeeze_value(self):
        """float，左手柄握紧的模拟量数值（0.0 ~ 1.0）。"""
        with self.left_squeeze_value_shared.get_lock():
            return self.left_squeeze_value_shared.value

    @property
    def left_controller_thumbstick_state(self):
        """bool，左摇杆是否被触摸或按下。"""
        with self.left_thumbstick_state_shared.get_lock():
            return self.left_thumbstick_state_shared.value

    @property
    def left_controller_thumbstick_value(self):
        """np.ndarray，shape (2,)，左摇杆的二维轴数值（x, y）。"""
        with self.left_thumbstick_value_shared.get_lock():
            return np.array(self.left_thumbstick_value_shared[:])

    @property
    def left_controller_aButton(self):
        """bool，左手柄的 A 键是否按下。"""
        with self.left_aButton_shared.get_lock():
            return self.left_aButton_shared.value

    @property
    def left_controller_bButton(self):
        """bool，左手柄的 B 键是否按下。"""
        with self.left_bButton_shared.get_lock():
            return self.left_bButton_shared.value

    @property
    def right_controller_trigger_state(self):
        """bool，右手柄扳机是否按下。"""
        with self.right_trigger_state_shared.get_lock():
            return self.right_trigger_state_shared.value

    @property
    def right_controller_trigger_value(self):
        """float，右手柄扳机的模拟量数值（0.0 ~ 1.0）。"""
        with self.right_trigger_value_shared.get_lock():
            return self.right_trigger_value_shared.value

    @property
    def right_controller_squeeze_state(self):
        """bool，右手柄握紧键是否按下。"""
        with self.right_squeeze_state_shared.get_lock():
            return self.right_squeeze_state_shared.value

    @property
    def right_controller_squeeze_value(self):
        """float，右手柄握紧的模拟量数值（0.0 ~ 1.0）。"""
        with self.right_squeeze_value_shared.get_lock():
            return self.right_squeeze_value_shared.value

    @property
    def right_controller_thumbstick_state(self):
        """bool，右摇杆是否被触摸或按下。"""
        with self.right_thumbstick_state_shared.get_lock():
            return self.right_thumbstick_state_shared.value

    @property
    def right_controller_thumbstick_value(self):
        """np.ndarray，shape (2,)，右摇杆的二维轴数值（x, y）。"""
        with self.right_thumbstick_value_shared.get_lock():
            return np.array(self.right_thumbstick_value_shared[:])

    @property
    def right_controller_aButton(self):
        """bool，右手柄的 A 键是否按下。"""
        with self.right_aButton_shared.get_lock():
            return self.right_aButton_shared.value

    @property
    def right_controller_bButton(self):
        """bool，右手柄的 B 键是否按下。"""
        with self.right_bButton_shared.get_lock():
            return self.right_bButton_shared.value