import numpy as np
from .televuer import TeleVuer
from dataclasses import dataclass, field

"""
（基）OpenXR 约定：y 向上、z 向后、x 向右。 
（基）机器人约定：z 向上、y 向左、x 向前。  

在（基）机器人约定下，人形机器人手臂的初始位姿约定：

    # （初始位姿）OpenXR 左臂位姿约定（手部追踪）：
        - x 轴从手腕指向中指。
        - y 轴从食指指向小指。
        - z 轴从手掌指向手背。

    # （初始位姿）OpenXR 右臂位姿约定（手部追踪）：
        - x 轴从手腕指向中指。
        - y 轴从小指指向食指。
        - z 轴从手掌指向手背。
  
    # （初始位姿）Unitree 人形机器人左臂 URDF 约定：
        - x 轴从手腕指向中指。
        - y 轴从手掌指向手背。
        - z 轴从小指指向食指。

    # （初始位姿）Unitree 人形机器人右臂 URDF 约定：
        - x 轴从手腕指向中指。
        - y 轴从手背指向手掌。 
        - z 轴从小指指向食指。

在（基）机器人约定下，人形机器人手部的初始位姿约定：

    # （初始位姿）OpenXR 左手位姿约定（手部追踪）：
        - x 轴从手腕指向中指。
        - y 轴从食指指向小指。
        - z 轴从手掌指向手背。

    # （初始位姿）OpenXR 右手位姿约定（手部追踪）：
        - x 轴从手腕指向中指。
        - y 轴从小指指向食指。
        - z 轴从手掌指向手背。

    # （初始位姿）Unitree 人形机器人左手 URDF 约定：
        - x 轴从手掌指向手背。 
        - y 轴从中指指向手腕。
        - z 轴从小指指向食指。

    # （初始位姿）Unitree 人形机器人右手 URDF 约定：
        - x 轴从手掌指向手背。 
        - y 轴从中指指向手腕。
        - z 轴从食指指向小指。 
    
注：TeleVuer 获取的所有原始数据都处于（基）OpenXR 约定下。 
     此外，手臂位姿数据（手部追踪）遵循（初始位姿）OpenXR 手臂位姿约定， 
     而手臂位姿数据（手柄追踪）直接遵循（初始位姿）Unitree 人形机器人手臂 URDF 约定（因此无需变换）。
     同时，所有原始数据都位于由 XR 设备里程计所定义的 WORLD（世界）坐标系中。

注：来自网站：https://registry.khronos.org/OpenXR/specs/1.1/man/html/openxr.html。
     你可以在其中找到如下与 **（初始位姿）OpenXR 左/右臂位姿约定** 相关的信息：
     “手腕关节位于手腕的枢轴点，当前臂保持不动而扭转手部时，该点的位置保持不变。 
     向后（+Z）方向平行于从手腕关节到中指掌骨关节的连线，并指向远离指尖的方向。 
     向上（+Y）方向指向手背，并垂直于手腕处的皮肤。 
     X 方向垂直于 Y 和 Z 方向，并满足右手定则。”
     注意：上述上下文当然处于 **（基）OpenXR 约定** 之下。

注：**Unitree 手臂/手部 URDF 初始位姿约定** 的信息来自 URDF 文件。
"""


def safe_mat_update(prev_mat, mat):
    # 当新矩阵非奇异（行列式 ≠ 0）时，返回上一矩阵和 False 标志。
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

def safe_rot_update(prev_rot_array, rot_array):
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True

# 常量
T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array([[1, 0, 0, 0],
                                           [0, 0,-1, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]])

T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0,-1, 0, 0],
                                            [0, 0, 0, 1]])

# UR3E机械臂坐标系转换矩阵
# 手柄坐标系：X轴指向手柄右侧，Y轴指向上方，Z轴指向前方
# UR3E末端坐标系：X轴指向前方，Y轴指向左侧，Z轴指向上方
T_TO_UR3E_RIGHT_ARM = np.array([[0, 0, 1, 0],  # 手柄Z轴 -> UR3E X轴（前方）
                                [1, 0, 0, 0],  # 手柄X轴 -> UR3E Y轴（左侧）
                                [0, 1, 0, 0],  # 手柄Y轴 -> UR3E Z轴（上方）
                                [0, 0, 0, 1]])

T_TO_UR3E_LEFT_ARM = np.array([[0, 0, 1, 0],   # 手柄Z轴 -> UR3E X轴（前方）
                               [-1, 0, 0, 0],  # 手柄X轴 -> UR3E Y轴（右侧，镜像）
                               [0, 1, 0, 0],   # 手柄Y轴 -> UR3E Z轴（上方）
                               [0, 0, 0, 1]])

T_TO_UNITREE_HAND = np.array([[0,  0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0,  0, 0, 1]])

T_ROBOT_OPENXR = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])

T_OPENXR_ROBOT = np.array([[ 0,-1, 0, 0],
                           [ 0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 0, 1]])

R_ROBOT_OPENXR = np.array([[ 0, 0,-1],
                           [-1, 0, 0],
                           [ 0, 1, 0]])

R_OPENXR_ROBOT = np.array([[ 0,-1, 0],
                           [ 0, 0, 1],
                           [-1, 0, 0]])

CONST_HEAD_POSE = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 1.5],
                            [0, 0, 1, -0.2],
                            [0, 0, 0, 1]])

# 机器人初始位置
CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15],
                                 [0, 1, 0, 1.13],
                                 [0, 0, 1, -0.3],
                                 [0, 0, 0, 1]])

CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15],
                                [0, 1, 0, 1.13],
                                [0, 0, 1, -0.3],
                                [0, 0, 0, 1]])

CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))

@dataclass
class TeleStateData:
    # 手部追踪
    left_pinch_state: bool = False         # 食指与拇指捏合时为 True
    left_squeeze_state: bool = False       # 手握拳时为 True
    left_squeeze_value: float = 0.0        # (0.0 ~ 1.0) 手握紧的程度
    right_pinch_state: bool = False        # 食指与拇指捏合时为 True
    right_squeeze_state: bool = False      # 手握拳时为 True
    right_squeeze_value: float = 0.0       # (0.0 ~ 1.0) 手握紧的程度

    # 手柄追踪
    left_trigger_state: bool = False       # 扳机被主动按下时为 True
    left_squeeze_ctrl_state: bool = False  # 握持键按下时为 True
    left_squeeze_ctrl_value: float = 0.0   # (0.0 ~ 1.0) 握持键扣动深度
    left_thumbstick_state: bool = False    # 摇杆按键按下时为 True
    left_thumbstick_value: np.ndarray = field(default_factory=lambda: np.zeros(2)) # 二维向量 (x, y)，已归一化
    left_aButton: bool = False             # A 键按下时为 True
    left_bButton: bool = False             # B 键按下时为 True
    right_trigger_state: bool = False      # 扳机被主动按下时为 True
    right_squeeze_ctrl_state: bool = False # 握持键按下时为 True
    right_squeeze_ctrl_value: float = 0.0  # (0.0 ~ 1.0) 握持键扣动深度
    right_thumbstick_state: bool = False   # 摇杆按键按下时为 True
    right_thumbstick_value: np.ndarray = field(default_factory=lambda: np.zeros(2)) # 二维向量 (x, y)，已归一化
    right_aButton: bool = False            # A 键按下时为 True
    right_bButton: bool = False            # B 键按下时为 True

@dataclass
class TeleData:
    head_pose: np.ndarray       # (4,4) 头部 SE(3) 位姿矩阵
    left_arm_pose: np.ndarray   # (4,4) 左臂 SE(3) 位姿
    right_arm_pose: np.ndarray  # (4,4) 右臂 SE(3) 位姿
    left_hand_pos: np.ndarray = None  # (25,3) 左手关节的三维位置
    right_hand_pos: np.ndarray = None # (25,3) 右手关节的三维位置
    left_hand_rot: np.ndarray  = None # (25,3,3) 左手关节的旋转矩阵
    right_hand_rot: np.ndarray = None # (25,3,3) 右手关节的旋转矩阵
    left_pinch_value: float = None    # float (1x.0 ~ 0.0) 食指与拇指之间的捏合距离
    right_pinch_value: float = None   # float (1x.0 ~ 0.0) 食指与拇指之间的捏合距离
    left_trigger_value: float = None  # float (10.0 ~ 0.0) 扳机扣动深度
    right_trigger_value: float = None # float (10.0 ~ 0.0) 扳机扣动深度
    tele_state: TeleStateData = field(default_factory=TeleStateData)


class TeleVuerWrapper:
    def __init__(self, binocular: bool, use_hand_tracking: bool, img_shape, img_shm_name, return_state_data: bool = True, return_hand_rot_data: bool = False,
                       cert_file = None, key_file = None, ngrok = False, webrtc = False, robot_type: str = "Unitree"):
        """
        TeleVuerWrapper 是 TeleVuer 类的封装，负责处理适用于机器人控制的 XR 设备数据套件。
        它使用指定参数初始化 TeleVuer 实例，并提供获取运动状态数据的方法。

        :param binocular: 布尔值，表示头部相机设备是否为双目。
        :param use_hand_tracking: 布尔值，表示使用手部追踪还是手柄追踪。
        :param img_shape: 待处理图像的形状。
        :param img_shm_name: 图像共享内存的名称。
        :param return_state: 布尔值，表示是否返回手部或手柄的状态。
        :param return_hand_rot: 布尔值，表示是否返回手部旋转数据。
        :param cert_file: 安全连接所用证书文件的路径。
        :param key_file: 安全连接所用密钥文件的路径。
        :param robot_type: 机器人类型（"Unitree"、"UR3E" 等），用于坐标变换。
        """
        self.use_hand_tracking = use_hand_tracking
        self.return_state_data = return_state_data
        self.return_hand_rot_data = return_hand_rot_data
        self.robot_type = robot_type
        self.tvuer = TeleVuer(binocular, use_hand_tracking, img_shape, img_shm_name, cert_file=cert_file, key_file=key_file,
                                ngrok=ngrok, webrtc=webrtc)
    
    def get_motion_state_data(self):
        """
        从 TeleVuer 实例获取处理后的运动状态数据。

        所有返回数据均已从 OpenXR 约定变换到（机器人 & Unitree）约定。
        """
        # 变量命名约定如下：
        # ┌────────────┬───────────────────────────┬──────────────────────────────────┬────────────────────────────────────┬────────────────────────────────────┐
        # │  左 / 右   │          Bxr              │              Brobot              │               IPxr                 │             IPunitree              │
        # │────────────│───────────────────────────│──────────────────────────────────│────────────────────────────────────│────────────────────────────────────│
        # │    侧别    │ （基）OpenXR 约定          │      （基）机器人约定             │ （初始位姿）OpenXR 约定            │ （初始位姿）Unitree 约定           │ 
        # └────────────┴───────────────────────────┴──────────────────────────────────┴────────────────────────────────────┴────────────────────────────────────┘
        # ┌───────────────────────────────────┬─────────────────────┐
        # │    world / arm / head / waist     │  arm / head / hand  │
        # │───────────────────────────────────│─────────────────────│
        # │             源坐标系              │       目标坐标系    │
        # └───────────────────────────────────┴─────────────────────┘

        # TeleVuer (Vuer) 获取的所有原始数据都处于（基）OpenXR 约定下。
        Bxr_world_head, head_pose_is_valid = safe_mat_update(CONST_HEAD_POSE, self.tvuer.head_pose)

        if self.use_hand_tracking:
            # “手臂”位姿数据遵循（基）OpenXR 约定以及（初始位姿）OpenXR 手臂约定。
            left_IPxr_Bxr_world_arm, left_arm_is_valid  = safe_mat_update(CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose)
            right_IPxr_Bxr_world_arm, right_arm_is_valid = safe_mat_update(CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose)

            # 更换基约定
            # 从（基）OpenXR 约定到（基）机器人约定：
            #   Brobot_Pose = T_{robot}_{openxr} * Bxr_Pose * T_{robot}_{openxr}^T  ==>
            #   Brobot_Pose = T_{robot}_{openxr} * Bxr_Pose * T_{openxr}_{robot}
            # 右乘 T_OPENXR_ROBOT = fast_mat_inv(T_ROBOT_OPENXR) 的原因：
            #   这是相似变换：B = PAP^{-1}，即 B ~ A
            #   举例说明：
            #   - 对于处于（基）机器人约定下的位姿数据 T_r，左乘 Brobot_Pose 意味着：
            #       Brobot_Pose * T_r  ==>  T_{robot}_{openxr} * PoseMatrix_openxr * T_{openxr}_{robot} * T_r
            #   - 首先，将 T_r 变换到（基）OpenXR 约定（T_{openxr}_{robot} 的作用）
            #   - 然后，在 OpenXR 约定下施加旋转 PoseMatrix_openxr（PoseMatrix_openxr 的作用）
            #   - 最后，变换回机器人约定（T_{robot}_{openxr} 的作用）
            #   - 这样在机器人约定下得到的旋转效果与在 OpenXR 约定下完全相同。
            Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT
            left_IPxr_Brobot_world_arm  = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
            right_IPxr_Brobot_world_arm = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT

            # 更换初始位姿约定
            # 从（初始位姿）OpenXR 手臂约定到（初始位姿）Unitree 人形机器人手臂 URDF 约定
            # 右乘 (T_TO_UNITREE_HUMANOID_LEFT_ARM) 的原因：绕自身 x 轴逆时针旋转 90 度。
            # 右乘 (T_TO_UNITREE_HUMANOID_RIGHT_ARM) 的原因：绕自身 x 轴顺时针旋转 90 度。
            left_IPunitree_Brobot_world_arm = left_IPxr_Brobot_world_arm @ (T_TO_UNITREE_HUMANOID_LEFT_ARM if left_arm_is_valid else np.eye(4))
            right_IPunitree_Brobot_world_arm = right_IPxr_Brobot_world_arm @ (T_TO_UNITREE_HUMANOID_RIGHT_ARM if right_arm_is_valid else np.eye(4))

            # 从 WORLD（世界）坐标系转换到 HEAD（头部）坐标系（仅调整平移）
            left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
            right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
            left_IPunitree_Brobot_head_arm[0:3, 3]  = left_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]
            right_IPunitree_Brobot_head_arm[0:3, 3] = right_IPunitree_Brobot_world_arm[0:3, 3] - Brobot_world_head[0:3, 3]

            # =====坐标原点偏移=====
            # IK 求解所用坐标系的原点位于 WAIST（腰部）关节电机附近。可使用 teleop/robot_control/robot_arm_ik.py 的 Unit_Test 进行可视化。
            # IPunitree_Brobot_head_arm 的坐标系原点为 HEAD（头部）。
            # 因此需要将 IPunitree_Brobot_head_arm 的原点从 HEAD 平移到 WAIST。
            left_IPunitree_Brobot_waist_arm = left_IPunitree_Brobot_head_arm.copy()
            right_IPunitree_Brobot_waist_arm = right_IPunitree_Brobot_head_arm.copy()
            left_IPunitree_Brobot_waist_arm[0, 3] +=0.15 # x 轴
            right_IPunitree_Brobot_waist_arm[0,3] +=0.15
            left_IPunitree_Brobot_waist_arm[2, 3] +=0.45 # z 轴
            right_IPunitree_Brobot_waist_arm[2,3] +=0.45

            # -----------------------------------手部位置----------------------------------------
            if left_arm_is_valid and right_arm_is_valid:
                # 齐次化，[xyz] 变为 [xyz1]
                #   np.concatenate([25,3]^T,(1,25)) ==> Bxr_world_hand_pos.shape 为 (4,25)
                # 此时在（基）OpenXR 约定下，Bxr_world_hand_pos 数据形式如下：
                #    [x0 x1 x2 ··· x23 x24]
                #    [y0 y1 y1 ··· y23 y24]
                #    [z0 z1 z2 ··· z23 z24]
                #    [ 1  1  1 ···  1    1]
                left_IPxr_Bxr_world_hand_pos  = np.concatenate([self.tvuer.left_hand_positions.T, np.ones((1, self.tvuer.left_hand_positions.shape[0]))])
                right_IPxr_Bxr_world_hand_pos = np.concatenate([self.tvuer.right_hand_positions.T, np.ones((1, self.tvuer.right_hand_positions.shape[0]))])

                # 更换基约定
                # 从（基）OpenXR 约定到（基）机器人约定
                # 对三维点而言只是基的更换，没有旋转，只有平移。因此无需右乘 fast_mat_inv(T_ROBOT_OPENXR)。
                left_IPxr_Brobot_world_hand_pos  = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_hand_pos
                right_IPxr_Brobot_world_hand_pos = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_hand_pos

                # 在（基）机器人约定下，从 WORLD 坐标系转换到 ARM（手臂）坐标系：
                #   Brobot_{world}_{arm}^T * Brobot_{world}_pos ==> Brobot_{arm}_{world} * Brobot_{world}_pos ==> Brobot_arm_hand_pos，此时基于手臂坐标系。
                left_IPxr_Brobot_arm_hand_pos  = fast_mat_inv(left_IPxr_Brobot_world_arm) @ left_IPxr_Brobot_world_hand_pos
                right_IPxr_Brobot_arm_hand_pos = fast_mat_inv(right_IPxr_Brobot_world_arm) @ right_IPxr_Brobot_world_hand_pos
                
                # 更换初始位姿约定
                # 从（初始位姿）XR 手部约定到（初始位姿）Unitree 人形机器人手部 URDF 约定：
                #   T_TO_UNITREE_HAND @ IPxr_Brobot_arm_hand_pos ==> IPunitree_Brobot_arm_hand_pos
                #   ((4,4) @ (4,25))[0:3, :].T ==> (4,25)[0:3, :].T ==> (3,25).T ==> (25,3)           
                # 此时在（初始位姿）Unitree 人形机器人手部 URDF 约定下，矩阵形式如下：
                #    [x0, y0, z0]
                #    [x1, y1, z1]
                #    ···
                #    [x23,y23,z23]
                #    [x24,y24,z24]
                left_IPunitree_Brobot_arm_hand_pos  = (T_TO_UNITREE_HAND @ left_IPxr_Brobot_arm_hand_pos)[0:3, :].T
                right_IPunitree_Brobot_arm_hand_pos = (T_TO_UNITREE_HAND @ right_IPxr_Brobot_arm_hand_pos)[0:3, :].T
            else:
                left_IPunitree_Brobot_arm_hand_pos  = np.zeros((25, 3))
                right_IPunitree_Brobot_arm_hand_pos = np.zeros((25, 3))

            # -----------------------------------手部旋转----------------------------------------
            if self.return_hand_rot_data:
                left_Bxr_world_hand_rot, left_hand_rot_is_valid  = safe_rot_update(CONST_HAND_ROT, self.tvuer.left_hand_orientations) # [25, 3, 3]
                right_Bxr_world_hand_rot, right_hand_rot_is_valid = safe_rot_update(CONST_HAND_ROT, self.tvuer.right_hand_orientations)

                if left_hand_rot_is_valid and right_hand_rot_is_valid:
                    left_Bxr_arm_hand_rot = np.einsum('ij,njk->nik', left_IPxr_Bxr_world_arm[:3, :3].T, left_Bxr_world_hand_rot)
                    right_Bxr_arm_hand_rot = np.einsum('ij,njk->nik', right_IPxr_Bxr_world_arm[:3, :3].T, right_Bxr_world_hand_rot)
                    # 更换基约定
                    left_Brobot_arm_hand_rot = np.einsum('ij,njk,kl->nil', R_ROBOT_OPENXR, left_Bxr_arm_hand_rot, R_OPENXR_ROBOT)
                    right_Brobot_arm_hand_rot = np.einsum('ij,njk,kl->nil', R_ROBOT_OPENXR, right_Bxr_arm_hand_rot, R_OPENXR_ROBOT)
                else:
                    left_Brobot_arm_hand_rot = left_Bxr_world_hand_rot
                    right_Brobot_arm_hand_rot = right_Bxr_world_hand_rot
            else:
                left_Brobot_arm_hand_rot = None
                right_Brobot_arm_hand_rot = None
            
            if self.return_state_data:
                hand_state = TeleStateData(
                    left_pinch_state=self.tvuer.left_hand_pinch_state,
                    left_squeeze_state=self.tvuer.left_hand_squeeze_state,
                    left_squeeze_value=self.tvuer.left_hand_squeeze_value,
                    right_pinch_state=self.tvuer.right_hand_pinch_state,
                    right_squeeze_state=self.tvuer.right_hand_squeeze_state,
                    right_squeeze_value=self.tvuer.right_hand_squeeze_value,
                )
            else:
                hand_state = None

            return TeleData(
                head_pose=Brobot_world_head,
                left_arm_pose=left_IPunitree_Brobot_waist_arm,
                right_arm_pose=right_IPunitree_Brobot_waist_arm,
                left_hand_pos=left_IPunitree_Brobot_arm_hand_pos,
                right_hand_pos=right_IPunitree_Brobot_arm_hand_pos,
                left_hand_rot=left_Brobot_arm_hand_rot,
                right_hand_rot=right_Brobot_arm_hand_rot,
                left_pinch_value=self.tvuer.left_hand_pinch_value * 100.0,
                right_pinch_value=self.tvuer.right_hand_pinch_value * 100.0,
                tele_state=hand_state
            )
        else:
            # 手柄位姿数据直接遵循（初始位姿）Unitree 人形机器人手臂 URDF 约定（因此无需变换）。
            left_IPunitree_Bxr_world_arm, left_arm_is_valid  = safe_mat_update(CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose)
            right_IPunitree_Bxr_world_arm, right_arm_is_valid = safe_mat_update(CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose)

            # 更换基约定
            Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT
            left_IPunitree_Brobot_world_arm  = T_ROBOT_OPENXR @ left_IPunitree_Bxr_world_arm @ T_OPENXR_ROBOT
            right_IPunitree_Brobot_world_arm = T_ROBOT_OPENXR @ right_IPunitree_Bxr_world_arm @ T_OPENXR_ROBOT
            
            # 在手柄模式下应用机器人特定的坐标变换
            if self.robot_type == "UR3E":
                # 在手柄模式下应用 UR3E 特定的变换
                left_IPunitree_Brobot_world_arm = left_IPunitree_Brobot_world_arm @ (T_TO_UR3E_LEFT_ARM if left_arm_is_valid else np.eye(4))
                right_IPunitree_Brobot_world_arm = right_IPunitree_Brobot_world_arm @ (T_TO_UR3E_RIGHT_ARM if right_arm_is_valid else np.eye(4))
            # 从 WORLD（世界）坐标系转换到 HEAD（头部）坐标系（仅调整平移）
            left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
            right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
            left_IPunitree_Brobot_head_arm[0:3, 3]  = left_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]
            right_IPunitree_Brobot_head_arm[0:3, 3] = right_IPunitree_Brobot_head_arm[0:3, 3] - Brobot_world_head[0:3, 3]

            # =====坐标原点偏移=====
            # IK 求解所用坐标系的原点位于 WAIST（腰部）关节电机附近。可使用 teleop/robot_control/robot_arm_ik.py 的 Unit_Test 进行查看。
            # IPunitree_Brobot_head_arm 的坐标系原点为 HEAD（头部）。 
            # 因此需要将 IPunitree_Brobot_head_arm 的原点从 HEAD 平移到 WAIST。
            left_IPunitree_Brobot_waist_arm = left_IPunitree_Brobot_head_arm.copy()
            right_IPunitree_Brobot_waist_arm = right_IPunitree_Brobot_head_arm.copy()
            left_IPunitree_Brobot_waist_arm[0, 3] +=0.15 # x 轴
            right_IPunitree_Brobot_waist_arm[0,3] +=0.15
            left_IPunitree_Brobot_waist_arm[2, 3] +=0.45 # z 轴
            right_IPunitree_Brobot_waist_arm[2,3] +=0.45
            # left_IPunitree_Brobot_waist_arm[1, 3] +=0.02 # y
            # right_IPunitree_Brobot_waist_arm[1,3] +=0.02

            # 返回数据
            if self.return_state_data:
                controller_state = TeleStateData(
                    left_trigger_state=self.tvuer.left_controller_trigger_state,
                    left_squeeze_ctrl_state=self.tvuer.left_controller_squeeze_state,
                    left_squeeze_ctrl_value=self.tvuer.left_controller_squeeze_value,
                    left_thumbstick_state=self.tvuer.left_controller_thumbstick_state,
                    left_thumbstick_value=self.tvuer.left_controller_thumbstick_value,
                    left_aButton=self.tvuer.left_controller_aButton,
                    left_bButton=self.tvuer.left_controller_bButton,
                    right_trigger_state=self.tvuer.right_controller_trigger_state,
                    right_squeeze_ctrl_state=self.tvuer.right_controller_squeeze_state,
                    right_squeeze_ctrl_value=self.tvuer.right_controller_squeeze_value,
                    right_thumbstick_state=self.tvuer.right_controller_thumbstick_state,
                    right_thumbstick_value=self.tvuer.right_controller_thumbstick_value,
                    right_aButton=self.tvuer.right_controller_aButton,
                    right_bButton=self.tvuer.right_controller_bButton,
                )
            else:
                controller_state = None

            return TeleData(
                head_pose=Brobot_world_head,
                left_arm_pose=left_IPunitree_Brobot_waist_arm,
                right_arm_pose=right_IPunitree_Brobot_waist_arm,
                left_trigger_value=10.0 - self.tvuer.left_controller_trigger_value * 10,
                right_trigger_value=10.0 - self.tvuer.right_controller_trigger_value * 10,
                tele_state=controller_state
            )