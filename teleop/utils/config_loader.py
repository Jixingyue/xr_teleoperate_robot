import yaml
import numpy as np
import os
from typing import Dict, Any

def load_robot_config(config_path: str = None) -> Dict[str, Any]:
    """
    从YAML配置文件加载机器人配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        Dict[str, Any]: 机器人配置字典
    """
    if config_path is None:
        # 获取当前文件所在目录的上级目录的config文件夹
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(parent_dir, 'config', 'robot_config.yml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 将YAML数据转换为Python字典格式
    return _convert_yaml_to_python_config(config_data)

def _convert_yaml_to_python_config(yaml_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将YAML数据转换为Python配置格式
    
    Args:
        yaml_data: YAML加载的数据
    Returns:
        Dict[str, Any]: 转换后的Python配置
    """
    config = {}
    
    for robot_type, robot_config in yaml_data.items():
        config[robot_type] = {}
        
        # 复制基本配置
        for key, value in robot_config.items():
            if key == 'end_effectors':
                # 处理末端执行器配置
                config[robot_type][key] = []
                for ee in value:
                    ee_config = {
                        'name': ee['name'],
                        'end_joint': ee['end_joint'],
                        'end_link': ee['end_link'],
                        'offset': np.array(ee['offset'])
                    }
                    config[robot_type][key].append(ee_config)
            elif key == 'arm_joints':
                config[robot_type][key] = []
                for joint in value:
                    joint_config = {
                        'name': joint['name'],
                        'index': joint['index']
                    }
                    config[robot_type][key].append(joint_config)
            elif key == 'gripper_joints':
                config[robot_type][key] = []
                for joint in value:
                    joint_config = {
                        'name': joint['name'],
                        'index': joint['index']
                    }
                    config[robot_type][key].append(joint_config)
            elif key == 'arm_scaling':
                config[robot_type][key] = {
                    'human_length': value['human_length'],
                    'robot_length': value['robot_length']
                }
            else:
                config[robot_type][key] = value
    if config[robot_type]['arm_scaling'] is None:
        config[robot_type]['arm_scaling'] = {
            'human_length': 0.45,
            'robot_length': 0.55
        }
    if config[robot_type].get('gripper_joints') is None:
        config[robot_type]['gripper_joints'] = []
    if config[robot_type].get('joints_to_lock') is None:
        config[robot_type]['joints_to_lock'] = []
    return config

def get_robot_config(robot_type: str, config_path: str = None) -> Dict[str, Any]:
    """
    获取指定机器人类型的配置
    
    Args:
        robot_type: 机器人类型
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 机器人配置
    """
    configs = load_robot_config(config_path)
    
    if robot_type not in configs:
        raise ValueError(f"不支持的机器人类型: {robot_type}")
    
    return configs[robot_type]

def list_available_robots(config_path: str = None) -> list:
    """
    列出所有可用的机器人类型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        list: 可用的机器人类型列表
    """
    configs = load_robot_config(config_path)
    return list(configs.keys()) 