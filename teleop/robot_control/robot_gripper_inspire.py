import numpy as np
import time
import threading
from multiprocessing import Process, Array, Value
from serial import Serial
import logging_mp
logger_mp = logging_mp.get_logger(__name__)
class SerialGripper:
    """串口夹爪控制类"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.serial = Serial(port, baudrate)
        print(f"Serial gripper initialized on {port}")
    
    def move_to_position(self, position):
        """
        控制夹爪移动到指定位置（0~1000），0最闭合，1000最张开。
        """
        b0b1 = b'\xEB\x90'
        b2 = b'\x01'  # 夹爪 ID
        b3 = b'\x03'  # 数据体长度
        b4 = b'\x54'  # 指令号 CMD_MC_SEEKPOS

        b5 = bytearray([position & 0x00FF])  # 开口度低位
        b6 = bytearray([position >> 8])      # 开口度高位

        checksum = (ord(b2) + ord(b3) + ord(b4) + b5[0] + b6[0]) & 0xFF
        b7 = bytearray([checksum])

        data = b0b1 + b2 + b3 + b4 + b5 + b6 + b7
        self.serial.write(data)
        return True
    
    def catch(self, speed=500, power=100):
        """抓取指令"""
        b0b1 = b'\xEB\x90'
        b2 = b'\x01'
        b3 = b'\x05'
        b4 = b'\x10'
        
        b5 = bytearray([speed & 0x00ff])  # 速度低位
        b6 = bytearray([speed >> 8])      # 速度高位
        b7 = bytearray([power & 0x00ff])  # 力度低位
        b8 = bytearray([power >> 8])      # 力度高位
        
        b9 = bytearray([(ord(b2) + ord(b3) + ord(b4) + ord(b5) + ord(b6) + ord(b7) + ord(b8)) & 0x00ff])
        data = b0b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9
        self.serial.write(data)
    
    def open(self, speed=5000):
        """打开夹爪"""
        b0b1 = b'\xEB\x90'
        b2 = b'\x01'
        b3 = b'\x03'
        b4 = b'\x11'

        b5 = bytearray([speed & 0x00ff])
        b6 = bytearray([speed >> 8])
        b7 = bytearray([(ord(b2) + ord(b3) + ord(b4) + ord(b5) + ord(b6)) & 0x00ff])
        data = b0b1 + b2 + b3 + b4 + b5 + b6 + b7
        self.serial.write(data)
        return True
    def get_state(self):
        """读取夹爪运行状态 (CMD 0x41)，自动拼帧"""
        # 清空串口缓冲区
        self.serial.reset_input_buffer()
        
        # 发送读取状态的指令帧
        cmd = bytes([0xEB, 0x90, 0x01, 0x01, 0x41, 0x43])
        self.serial.write(cmd)
        
        # 等待响应并读取完整帧
        time.sleep(0.1)  # 添加适当延迟确保数据接收完整
        
        # 先读取帧头
        header = self.serial.read(2)
        if len(header) < 2 or header[0] != 0xEE or header[1] != 0x16:
            # print("⚠️ 帧头错误或接收不完整")
            return self.get_state()
        
        # 读取ID号和数据体长度
        id_and_length = self.serial.read(2)
        if len(id_and_length) < 2:
            print("⚠️ 无法读取ID和数据体长度")
            return self.get_state()
        
        dev_id = id_and_length[0]
        data_length = id_and_length[1]
        
        # 读取剩余数据（指令号 + 数据内容 + 校验和）
        remaining_data = self.serial.read(data_length + 1)  # +1 for checksum
        if len(remaining_data) < data_length + 1:
            print("⚠️ 数据接收不完整")
            return self.get_state()
        
        # 组合完整帧
        frame = header + id_and_length + remaining_data
        
        # 验证数据体长度（根据协议，状态查询响应应该是8字节数据体）
        # 但实际设备可能返回不同的长度，我们先接受并解析
        if data_length != 0x08:
            # print(f"⚠️ 数据体长度与预期不符: 期望0x08, 实际{data_length:02X}，继续解析...")
            return self.get_state()
        
        # 校验和验证（从ID号开始到数据内容结束）
        checksum = sum(frame[2:-1]) & 0xFF
        if checksum != frame[-1]:
            # print(f"⚠️ 校验和错误: 计算值={checksum:02X}, 接收值={frame[-1]:02X}")
            return self.get_state()
        
        # 解析数据字段
        cmd_code = frame[4]
        
        # 根据数据体长度解析不同的字段
        if data_length >= 8:
            # 完整状态响应
            run_state = frame[5]
            fault_code = frame[6]
            temperature = frame[7]
            position = frame[8] | (frame[9] << 8)  # 小端格式
            force_setting = frame[10] | (frame[11] << 8)  # 小端格式
        else:
            # 简化响应，只包含基本状态
            run_state = frame[5] if data_length >= 2 else 0
            fault_code = 0
            temperature = 0
            position = 0
            force_setting = 0

        # 运行状态映射表
        run_state_map = {
            0x01: "张开到最大且空闲",
            0x02: "闭合到最小且空闲",
            0x03: "停止且空闲",
            0x04: "正在闭合",
            0x05: "正在张开",
            0x06: "闭合遇到力控停止",
            0x54: "位置移动指令响应",  # 添加对位置移动指令响应的处理
            0x10: "抓取指令响应",      # 添加对抓取指令响应的处理
            0x11: "打开指令响应"       # 添加对打开指令响应的处理
        }

        # 故障码解析
        fault_list = []
        if fault_code & 0x01: fault_list.append("堵转")
        if fault_code & 0x02: fault_list.append("过温")
        if fault_code & 0x04: fault_list.append("过流")
        if fault_code & 0x08: fault_list.append("驱动器运行故障")
        if fault_code & 0x10: fault_list.append("内部通信故障")
        if not fault_list:
            fault_list.append("正常")

        return {
            "device_id": dev_id,
            "run_state": run_state_map.get(run_state, f"未知状态({run_state})"),
            "fault": fault_list,
            "temperature": temperature,
            "position": position/1000,
            "force_setting": force_setting
        }



    def close(self):
        """关闭串口连接"""
        self.serial.close()
        return
        

class Inspire_Gripper_Controller:
    def __init__(self, left_gripper_value=None, right_gripper_value=None, dual_gripper_data_lock=None, 
                 dual_gripper_state_array=None, dual_gripper_action_array=None, fps=100.0, Unit_Test=False):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        
        # 初始化串口夹爪
        if right_gripper_value is not None and left_gripper_value is None:
            # 只有右手夹爪
            self.right_gripper = SerialGripper('/dev/ttyUSB0')
            self.left_gripper = None
        elif left_gripper_value is not None and right_gripper_value is not None:
            # 双手夹爪
            self.left_gripper = SerialGripper('/dev/ttyUSB0')
            self.right_gripper = SerialGripper('/dev/ttyUSB1')
        else:
            raise ValueError("left_gripper_value and right_gripper_value must be not None at least one")
        
        # 保存共享数组引用
        self.left_gripper_state_value  = Value('d',1.0,lock=True)
        self.right_gripper_state_value = Value('d',1.0,lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        
        gripper_control_process = Process(target=self.control_process, args=(left_gripper_value, right_gripper_value,  self.left_gripper_state_value, self.right_gripper_state_value,
                                                                          dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array))
        gripper_control_process.daemon = True
        gripper_control_process.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def ctrl_dual_gripper(self, left_position, right_position):
        """
        控制左右手夹爪位置
        position: 0.0-1.0，0.0表示完全闭合，1.0表示完全张开
        """
        def position_to_gripper(pos):
            """将0-1的位置值转换为夹爪的0-10位置"""
            return int(pos * 1000)
        
        # 控制左手夹爪
        if self.left_gripper is not None:
            left_gripper_pos = position_to_gripper(left_position)
            self.left_gripper.move_to_position(left_gripper_pos)
        
        # 控制右手夹爪
        if self.right_gripper is not None:
            right_gripper_pos = position_to_gripper(right_position)
            self.right_gripper.move_to_position(right_gripper_pos)
    
    def control_process(self, left_gripper_value, right_gripper_value, left_gripper_state_value, right_gripper_state_value,
                              dual_gripper_data_lock = None, dual_gripper_state_array = None, dual_gripper_action_array = None):
        self.running = True
        # 初始化夹爪位置为半开状态
        left_target_q = 1
        right_target_q = 1

        try:
            while self.running:
                start_time = time.time()
                
                # 获取夹爪控制值
                if left_gripper_value is not None:
                    with left_gripper_value.get_lock():
                        left_target_q = left_gripper_value.value
                
                if right_gripper_value is not None:
                    with right_gripper_value.get_lock():
                        right_target_q = right_gripper_value.value
                
                # 确保位置值在有效范围内
                left_target_q = np.clip(left_target_q, 0.0, 1.0)
                right_target_q = np.clip(right_target_q, 0.0, 1.0)
                
                # 更新状态和动作数组
                if dual_gripper_state_array is not None and dual_gripper_action_array is not None:
                    with dual_gripper_data_lock:
                        # 状态数组：[左手位置, 右手位置]
                        dual_gripper_state_array[0] = left_gripper_state_value.value
                        dual_gripper_state_array[1] = right_gripper_state_value.value
                        
                        # 动作数组：[左手目标位置, 右手目标位置]
                        dual_gripper_action_array[0] = left_target_q
                        dual_gripper_action_array[1] = right_target_q
                
                # 控制夹爪
                self.ctrl_dual_gripper(left_target_q, right_target_q)
                
                # 控制频率
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger_mp.error(f"Error in control_process: {e}")
        finally:
            # 关闭串口连接
            if self.left_gripper is not None:
                self.left_gripper.close()
            if self.right_gripper is not None:
                self.right_gripper.close()
            logger_mp.info("Inspire_Controller has been closed.")
    def _subscribe_gripper_state(self):
        while True:
            if self.left_gripper is not None:   
                self.left_gripper_state_value.value = self.left_gripper.get_state()["position"]
            if self.right_gripper is not None:
                self.right_gripper_state_value.value = self.right_gripper.get_state()["position"]
            # print(f"左手状态：{self.left_gripper_state_value.value}, 右手状态：{self.right_gripper_state_value.value}")
            time.sleep(0.002)
if __name__ == "__main__":
    print("=== 夹爪通信测试程序 ===")
    # 然后测试 SerialGripper 类
    print("\n2. 测试 SerialGripper 类...")
    try:
        gripper = SerialGripper('/dev/ttyUSB0')
        
        # 测试移动到不同位置
        test_positions = [1000]
        for pos in test_positions:
            print(f"移动到位置 {pos}...")
            gripper.move_to_position(pos)
            time.sleep(1)
            
            # 尝试读取状态
            try:
                state = gripper.get_state()
                print(f"位置 {pos} 后的状态: {state}")
            except Exception as e:
                print(f"读取状态失败: {e}")
        
        gripper.close()
        print("✓ SerialGripper 测试完成")
        
    except Exception as e:
        print(f"✗ SerialGripper 测试失败: {e}")
    