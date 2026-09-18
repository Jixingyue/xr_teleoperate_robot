from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from typing import Union

import numpy as np
import yaml
import os

from . import yourdfpy as urdf
from .kinematics_adaptor import MimicJointKinematicAdaptor
from .optimizer_utils import LPFilter
from .robot_wrapper import RobotWrapper
from .seq_retarget import SeqRetargeting
from .yourdfpy import DUMMY_JOINT_NAMES


@dataclass
class RetargetingConfig:
    type: str
    urdf_path: str
    target_joint_names: Optional[List[str]] = None

    # 是否为机器人根部添加自由关节。自由关节使机器人手可以在空间中自由移动
    add_dummy_free_joint: bool = False

    # DexPilot 重定向相关
    # 机器人手上对应人手手腕的连杆
    wrist_link_name: Optional[str] = None
    # DexPilot 重定向连杆名称
    finger_tip_link_names: Optional[List[str]] = None
    target_link_human_indices_dexpilot: Optional[np.ndarray] = None

    # Position 重定向连杆名称
    target_link_names: Optional[List[str]] = None
    target_link_human_indices_position: Optional[np.ndarray] = None

    # Vector 重定向连杆名称
    target_origin_link_names: Optional[List[str]] = None
    target_task_link_names: Optional[List[str]] = None
    target_link_human_indices_vector: Optional[np.ndarray] = None

    # 仅用于 vector 重定向的缩放系数
    # 例如，Allegro 比普通人手大 1.6 倍，则该缩放系数应为 1.6
    scaling_factor: float = 1.0

    # 低通滤波器
    low_pass_alpha: float = 0.1

    # 优化参数
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # DexPilot 优化器参数
    project_dist: float = 0.03
    escape_dist: float = 0.05

    # 关节限位标记
    has_joint_limits: bool = True

    # mimic 关节标记
    ignore_mimic_joint: bool = False

    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = "./"

    def __post_init__(self):
        # 重定向类型检查
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")

        # Vector 重定向需要: target_origin_link_names + target_task_link_names
        # Position 重定向需要: target_link_names
        if self.type == "vector":
            if self.target_origin_link_names is None or self.target_task_link_names is None:
                raise ValueError(f"Vector retargeting requires: target_origin_link_names + target_task_link_names")
            if len(self.target_task_link_names) != len(self.target_origin_link_names):
                raise ValueError(f"Vector retargeting origin and task links dim mismatch")
            if self.target_link_human_indices_vector.shape != (2, len(self.target_origin_link_names)):
                raise ValueError(f"Vector retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices_vector is None:
                raise ValueError(f"Vector retargeting requires: target_link_human_indices_vector")

        elif self.type == "position":
            if self.target_link_names is None:
                raise ValueError(f"Position retargeting requires: target_link_names")
            self.target_link_human_indices_position = self.target_link_human_indices_position.squeeze()
            if self.target_link_human_indices_position.shape != (len(self.target_link_names),):
                raise ValueError(f"Position retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices_position is None:
                raise ValueError(f"Position retargeting requires: target_link_human_indices_position")

        elif self.type == "dexpilot":
            if self.finger_tip_link_names is None or self.wrist_link_name is None:
                raise ValueError(f"Position retargeting requires: finger_tip_link_names + wrist_link_name")
            if self.target_link_human_indices_dexpilot is not None:
                print(
                    "\033[33m",
                    "Target link human indices is provided in the DexPilot retargeting config, which is uncommon.\n"
                    "If you do not know exactly how it is used, please leave it to None for default.\n"
                    "\033[00m",
                )

        # URDF 路径检查
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: Union[str, Path]):
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path], override: Optional[Dict] = None):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], override: Optional[Dict] = None):
        if "target_link_human_indices_position" in cfg:
            cfg["target_link_human_indices_position"] = np.array(cfg["target_link_human_indices_position"])
        if "target_link_human_indices_vector" in cfg:
            cfg["target_link_human_indices_vector"] = np.array(cfg["target_link_human_indices_vector"])
        if "target_link_human_indices_dexpilot" in cfg:
            cfg["target_link_human_indices_dexpilot"] = np.array(cfg["target_link_human_indices_dexpilot"])

        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> SeqRetargeting:
        from .optimizer import (
            VectorOptimizer,
            PositionOptimizer,
            DexPilotOptimizer,
        )
        import tempfile

        # 使用 yourdfpy 处理 URDF 以便更好地查找文件路径
        robot_urdf = urdf.URDF.load(
            self.urdf_path, add_dummy_free_joints=self.add_dummy_free_joint, build_scene_graph=False
        )
        urdf_name = self.urdf_path.split(os.path.sep)[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # 加载 pinocchio 模型
        robot = RobotWrapper(temp_path)

        # 将 6D dummy 关节加入目标关节名称，使其同样参与优化
        if self.add_dummy_free_joint and self.target_joint_names is not None:
            self.target_joint_names = DUMMY_JOINT_NAMES + self.target_joint_names
        joint_names = self.target_joint_names if self.target_joint_names is not None else robot.dof_joint_names

        if self.type == "position":
            optimizer = PositionOptimizer(
                robot,
                joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices_position,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "vector":
            optimizer = VectorOptimizer(
                robot,
                joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices_vector,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "dexpilot":
            optimizer = DexPilotOptimizer(
                robot,
                joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                target_link_human_indices=self.target_link_human_indices_dexpilot,
                scaling=self.scaling_factor,
                project_dist=self.project_dist,
                escape_dist=self.escape_dist,
            )
        else:
            raise RuntimeError()

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        # 解析 mimic 关节并为优化器设置运动学适配器
        has_mimic_joints, source_names, mimic_names, multipliers, offsets = parse_mimic_joint(robot_urdf)
        if has_mimic_joints and not self.ignore_mimic_joint:
            adaptor = MimicJointKinematicAdaptor(
                robot,
                target_joint_names=joint_names,
                source_joint_names=source_names,
                mimic_joint_names=mimic_names,
                multipliers=multipliers,
                offsets=offsets,
            )
            optimizer.set_kinematic_adaptor(adaptor)
            print(
                "\033[34m",
                "Mimic joint adaptor enabled. The mimic joint tags in the URDF will be considered during retargeting.\n"
                "To disable mimic joint adaptor, consider setting ignore_mimic_joint=True in the configuration.",
                "\033[39m",
            )

        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
        )
        return retargeting


def get_retargeting_config(config_path: Union[str, Path]) -> RetargetingConfig:
    config = RetargetingConfig.load_from_file(config_path)
    return config


def parse_mimic_joint(robot_urdf: urdf.URDF) -> Tuple[bool, List[str], List[str], List[float], List[float]]:
    mimic_joint_names = []
    source_joint_names = []
    multipliers = []
    offsets = []
    for name, joint in robot_urdf.joint_map.items():
        if joint.mimic is not None:
            mimic_joint_names.append(name)
            source_joint_names.append(joint.mimic.joint)
            multipliers.append(joint.mimic.multiplier)
            offsets.append(joint.mimic.offset)

    return len(mimic_joint_names) > 0, source_joint_names, mimic_joint_names, multipliers, offsets
