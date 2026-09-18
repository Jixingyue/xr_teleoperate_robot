import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None, wrist_img_shape = None, wrist_img_shm_name = None, 
                       image_show = False, server_address = "192.168.123.164", port = 5555, Unit_Test = False):
        """
        tv_img_shape: 用户期望的头部相机分辨率形状 (H, W, C)，应与图像服务端的输出保持一致。

        tv_img_shm_name: shared memory/共享内存名称，用于方便地跨进程向 Vuer 传输图像。

        wrist_img_shape: 用户期望的腕部相机分辨率形状 (H, W, C)，应与 tv_img_shape 保持相同形状。

        wrist_img_shm_name: shared memory/共享内存名称，用于方便地传输图像。
        
        image_show: 是否实时显示接收到的图像。

        server_address: 运行图像服务器脚本的 IP 地址。

        port: 要绑定的端口号，应与图像服务器保持一致。

        Unit_Test: 当服务器和客户端均为 True 时，可用于测试图像传输 latency/延迟、\
                   网络 jitter/抖动、丢帧率等信息。
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = self.tv_image_shm.buf)
            self.tv_enable_shm = True
        
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # 性能评估参数
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self._frame_count = 0  # 已接收的总帧数
        self._last_frame_id = -1  # 最近一次接收到的帧 ID

        # 使用时间窗口实时计算 fps/帧率
        self._time_window = 1.0  # 时间窗口大小（单位：秒）
        self._frame_times = deque()  # 时间窗口内接收到的各帧时间戳

        # 数据传输质量指标
        self._latencies = deque()  # 时间窗口内各帧的 latency/延迟
        self._lost_frames = 0  # 总 lost frame/丢帧数
        self._total_frames = 0  # 根据帧 ID 推算的预期总帧数

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # 更新延迟
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # 移除时间窗口之外的延迟数据
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # 更新帧时间戳
        self._frame_times.append(receive_time)
        # 移除时间窗口之外的时间戳
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # 更新帧计数，用于丢帧计算
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                logger_mp.info(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                logger_mp.warning(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # 计算实时 fps/帧率
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # 计算延迟指标
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # 计算丢帧率
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            logger_mp.info(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _close(self):
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        logger_mp.info("Image client has been closed.")

    
    def receive_process(self):
        # 创建 ZeroMQ/ZMQ 上下文和套接字
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        logger_mp.info("Image client has started, waiting to receive data...")
        try:
            while self.running:
                # 接收消息
                message = self._socket.recv()
                receive_time = time.time()

                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        # 尝试解析消息头和图像数据
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        logger_mp.warning(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    # 无消息头，整条消息均为图像数据
                    jpg_bytes = message
                # 解码图像
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    logger_mp.warning("[Image Client] Failed to decode image.")
                    continue
                if self.tv_enable_shm:
                    np.copyto(self.tv_img_array, np.array(current_image[:, :self.tv_img_shape[1]]))
                
                if self.wrist_enable_shm:
                    np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1]:]))
                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow('Image Client Stream', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            logger_mp.info("Image client interrupted by user.")
        except Exception as e:
            logger_mp.warning(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

if __name__ == "__main__":
    # 示例1
    # tv_img_shape = (720, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(image_show = False,server_address='192.168.123.116',tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # # img_client.receive_process()
    # # 启动图像接收线程
    # import threading
    # image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    # image_receive_thread.start()
    # # 等待第一帧图像数据
    # print("等待图像数据...")
    # while True:
    #     # 检查图像数据是否有效（不是全零或全黑）
    #     if np.any(img_array):
    #         # 检查图像是否包含有效内容（简单的非零像素检查）
    #         if np.mean(img_array) > 1.0:  # 如果平均像素值大于1，认为有有效图像
    #             tv_resized_image = cv2.resize(img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
    #             cv2.imshow("teleop view", tv_resized_image)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #         else:
    #             print("等待有效图像数据...")
    #     else:
    #         print("等待图像数据...")
    #     time.sleep(0.2)
    
    # # 清理资源
    # cv2.destroyAllWindows()
    # img_shm.close()
    # img_shm.unlink()

    # 示例2
    # 初始化客户端并开启性能评估
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) # 本地测试
    client = ImageClient(image_show = True, server_address='192.168.123.116', Unit_Test=False) # 部署测试
    client.receive_process()