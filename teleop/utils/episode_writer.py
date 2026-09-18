import os
import cv2
import json
import datetime
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class EpisodeWriter():
    def __init__(self, task_dir, frequency=30, image_size=[640, 480], rerun_log = True):
        """
        image_size: [宽度, 高度]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc()

        self.is_available = True  # 表示该类当前是否可接受新的操作
        # 初始化队列和工作线程
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # 用于标记 save_episode 何时被触发的标志位
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   ['kLeftShoulderPitch' ,'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow', 'kLeftWristRoll', 'kLeftWristPitch', 'kLeftWristyaw'],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }
    def text_desc(self):
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "Pick up the cup from the table and place it in another position. The operation should be smooth and the water in the cup should not spill out",
            "steps":"step1: searching for cups. step2: go to the target location. step3: pick up the cup",
        }

 
    def create_episode(self):
        """
        创建一个新的 episode。
        Returns:
            bool: episode 创建成功返回 True，否则返回 False。
        Note:
            一旦创建成功，该函数只有在 save_episode 完成其保存任务后才能再次使用。
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # 该类不可用时返回 False

        # 重置 episode 相关数据并创建所需的目录
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # episode 创建后，该类被标记为不可用，直到 episode 成功保存
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # episode 创建成功时返回 True
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None):
        # item ID 自增
        self.item_id += 1
        # 创建 item 数据字典
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        # 将 item 数据放入队列
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # 处理队列中的 item
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # 检查 save_episode 是否被触发
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # 保存图像
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # 保存 depth/深度图
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info(f"Failed to save depth image.")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # 保存音频
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # 更新 episode 数据
        self.episode_data.append(item_data)

        # 如有需要，记录数据日志
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        触发保存操作。此处设置保存标志位，由 process_queue 线程负责处理。
        """
        self.need_save = True  # 设置保存标志位
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        将 episode 数据保存到 JSON 文件。
        """
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        with open(self.json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.need_save = False     # 重置保存标志位
        self.is_available = True   # 保存完成后将该类标记为可用
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        停止工作线程并确保所有任务均已完成。
        """
        self.item_data_queue.join()
        if not self.is_available:  # 如果 self.is_available 为 False，说明仍有数据未保存。
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()