from abc import abstractmethod
from typing import List

import numpy as np

from .robot_wrapper import RobotWrapper


class KinematicAdaptor:
    def __init__(self, robot: RobotWrapper, target_joint_names: List[str]):
        self.robot = robot
        self.target_joint_names = target_joint_names

        # 索引映射
        self.idx_pin2target = np.array([robot.get_joint_index(n) for n in target_joint_names])

    @abstractmethod
    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        针对不同的运动学约束对关节位置进行适配。
        注意该 qpos 的关节顺序与 pinocchio 一致
        Args:
            qpos: pinocchio 的 qpos

        Returns: 与输入形状相同的适配后 qpos

        """
        pass

    @abstractmethod
    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        针对不同的运动学应用对雅可比矩阵进行适配。
        注意该 Jacobian 的关节顺序与 pinocchio 一致
        Args:
            jacobian: 原始雅可比矩阵

        Returns: 与输入形状相同的适配后雅可比矩阵

        """
        pass


class MimicJointKinematicAdaptor(KinematicAdaptor):
    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        source_joint_names: List[str],
        mimic_joint_names: List[str],
        multipliers: List[float],
        offsets: List[float],
    ):
        super().__init__(robot, target_joint_names)

        self.multipliers = np.array(multipliers)
        self.offsets = np.array(offsets)

        # 关节名称检查
        union_set = set(mimic_joint_names).intersection(set(target_joint_names))
        if len(union_set) > 0:
            raise ValueError(
                f"Mimic joint should not be one of the target joints.\n"
                f"Mimic joints: {mimic_joint_names}.\n"
                f"Target joints: {target_joint_names}\n"
                f"You need to specify the target joint names explicitly in your retargeting config"
                f" for robot with mimic joint constraints: {target_joint_names}"
            )

        # 在 pinocchio 中的索引
        self.idx_pin2source = np.array([robot.get_joint_index(name) for name in source_joint_names])
        self.idx_pin2mimic = np.array([robot.get_joint_index(name) for name in mimic_joint_names])

        # 在输出结果中的索引
        self.idx_target2source = np.array([self.target_joint_names.index(n) for n in source_joint_names])

        # 维度检查
        len_source, len_mimic = self.idx_target2source.shape[0], self.idx_pin2mimic.shape[0]
        len_mul, len_offset = self.multipliers.shape[0], self.offsets.shape[0]
        if not (len_mimic == len_source == len_mul == len_offset):
            raise ValueError(
                f"Mimic joints setting dimension mismatch.\n"
                f"Source joints: {len_source}, mimic joints: {len_mimic}, multiplier: {len_mul}, offset: {len_offset}"
            )
        self.num_active_joints = len(robot.dof_joint_names) - len_mimic

        # 唯一性检查
        if len(mimic_joint_names) != len(np.unique(mimic_joint_names)):
            raise ValueError(f"Redundant mimic joint names: {mimic_joint_names}")

    def forward_qpos(self, pin_qpos: np.ndarray) -> np.ndarray:
        mimic_qpos = pin_qpos[self.idx_pin2source] * self.multipliers + self.offsets
        pin_qpos[self.idx_pin2mimic] = mimic_qpos
        return pin_qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        target_jacobian = jacobian[..., self.idx_pin2target]
        mimic_joint_jacobian = jacobian[..., self.idx_pin2mimic] * self.multipliers

        for i, index in enumerate(self.idx_target2source):
            target_jacobian[..., index] += mimic_joint_jacobian[..., i]
        return target_jacobian
