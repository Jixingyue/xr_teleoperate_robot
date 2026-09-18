# teleop —— XR 头显遥操作机械臂（小白上手指南）

> 本目录是整个项目的**核心代码目录**。如果你是第一次接触这个项目，建议先通读本文档，再动手运行。
>
> 项目根目录的 [README.md](../README.md) / [README_zh-CN.md](../README_zh-CN.md) 是宇树科技官方说明（针对 G1/H1 人形机器人）；本定制版在官方代码基础上**扩展了工业机械臂（遨博 i5、UR3E、Franka Panda）和 PyBullet 仿真**的支持。根目录的 [适配真实机械臂.md](../适配真实机械臂.md) 和 [仿真中机械臂适配md](../仿真中机械臂适配md) 是两份非常好的进阶开发文档。

---

## 一、这个项目是做什么的？

用 **XR 头显**（Meta Quest 3 / Apple Vision Pro / PICO 4 Ultra 等）遥操作机械臂：

- 戴上头显，你能以**机械臂第一人称视角**看到相机画面；
- 转动头部、移动手柄（或徒手做手势），程序实时把你的**手腕位姿**换算成机械臂关节角度（逆运动学，IK）；
- 捏合手指或扣动扳机，即可控制**灵巧手 / 夹爪**的开合；
- 还可以把整个过程的图像、关节状态、动作**录制成数据集**（用于模仿学习）。

### 数据是怎么流动的？（理解这张图，就理解了整个项目）

```
┌─────────────┐   ① 头/手腕/手部的位姿（WebXR, wss 加密 websocket, 端口 8012）
│  XR 头显     │ ────────────────────────────────────────────────┐
│ (浏览器/Vuer) │ ◄────────────────────────────────────────────────┘
└─────────────┘   ④ 机械臂第一人称相机画面回传给头显
        ▲
        │ TeleVuerWrapper（坐标变换：OpenXR 坐标系 → 机器人坐标系）
        ▼
┌──────────────────────────────────────────────┐
│              主程序（60Hz 循环）                │
│  手腕 4×4 位姿 ──► 逆运动学 IK  ──► 6/7 个关节角 │
│  手部 25 个关键点 ──► 灵巧手重定向 ──► 手指关节角 │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────┐    Socket / 厂商 SDK / DDS     ┌──────────────┐
│ 机械臂控制器   │ ────────────────────────────► │ 真实机械臂     │
└──────────────┘                                └──────────────┘

┌──────────────┐   ③ ZeroMQ 发布图像（tcp 端口 5555）
│ 相机(image_   │ ──────────────────────────────► 主程序 image_client（共享内存）
│ server，相机端)│                                                 │
└──────────────┘                                                 ▼
                                                        ② 送给 televuer 回传头显
```

两条网络链路要分清：

| 链路 | 端口 | 协议 | 作用 |
|------|------|------|------|
| 头显 ↔ 主程序 | **8012** | wss（https + websocket） | 传你的动作、回传相机画面 |
| 相机端 → 主程序 | **5555** | ZeroMQ（tcp） | 传输 JPEG 图像（仿真时不需要） |

---

## 二、目录结构说明

只列出与理解、运行项目相关的文件：

```
xr_teleoperate_robot/
│
├── teleop/                        ← 本目录，所有遥操作代码
│   │
│   ├── teleop_hand_and_arm.py        【入口①】宇树原版：G1/H1 人形双臂遥操作
│   ├── teleop_hand_and_arm_aubo.py   【入口②】定制：真机单臂 + Inspire 灵巧手（默认遨博 i5）
│   ├── teleop_gripper_and_arm.py     【入口③】定制：真机单臂 + 夹爪
│   ├── teleop_hand_and_arm_sim.py    【入口④】定制：PyBullet 仿真（无需任何真机，新手首选）
│   │
│   ├── image_server/                 图像传输服务
│   │   ├── image_server.py           相机端运行：读取 OpenCV / RealSense 相机并通过 ZMQ 发布
│   │   └── image_client.py           主机端运行：接收图像，写入共享内存供主程序使用
│   │
│   ├── televuer/                     子模块（pip install -e . 安装为 televuer 包）
│   │   ├── cert.pem / key.pem        HTTPS 自签名证书（已生成，头显连接必需）
│   │   └── src/televuer/
│   │       ├── televuer.py           底层封装 Vuer：起 https/wss 服务，采集 XR 数据、推图像
│   │       └── tv_wrapper.py          ★ 重点：TeleVuerWrapper，坐标系变换 + 数据结构化(TeleData)
│   │
│   ├── robot_control/                机器人控制
│   │   ├── robot_arm_basic.py        ★ RobotArmController 抽象基类（线程模型、限位、回零）
│   │   ├── universal_robot_arm.py    ★ 定制：UR3E / 遨博i5 控制器具体实现 + 工厂函数
│   │   ├── universal_robot_arm_ik.py ★ 定制：通用逆运动学解算器（Pinocchio + CasADi）
│   │   ├── aubo_sdk.py               遨博官方 Python SDK（厂商代码，不要改）
│   │   ├── robot_arm.py / robot_arm_ik.py          宇树 G1/H1 双臂控制器与 IK（官方原版）
│   │   ├── robot_hand_inspire.py     Inspire 灵巧手控制（DDS 通信）
│   │   ├── robot_gripper_inspire.py  串口电动夹爪控制（/dev/ttyUSB*）
│   │   ├── robot_hand_unitree.py     宇树 Dex3 灵巧手 / Dex1 夹爪控制（官方原版）
│   │   ├── robot_hand_brainco.py     BrainCo 灵巧手控制（官方原版）
│   │   ├── hand_retargeting.py       灵巧手重定向配置封装（调用 dex-retargeting 库）
│   │   └── dex-retargeting/          子模块：开源灵巧手重定向算法库（pip install -e .）
│   │
│   └── utils/                        工具模块
│       ├── config_loader.py          ★ 读取 config/robot_config.yml
│       ├── episode_writer.py         数据集录制（图像 + states + actions，供模仿学习）
│       ├── weighted_moving_filter.py 关节角度加权滑动滤波（让动作更平滑）
│       ├── performance_monitor.py    性能监控（帧率、各环节耗时统计）
│       ├── rerun_visualizer.py       用 Rerun 可视化录制过程
│       └── sim_state_topic.py        Isaac 仿真状态订阅（仅宇树原版仿真用）
│
├── config/
│   └── robot_config.yml           ★ 定制机器人的配置：URDF 路径、关节名/索引、关节限位、臂长缩放
│
├── assets/                        各机器人 URDF 模型与网格
│   ├── franka_description/        Franka Panda（PyBullet 仿真默认使用）
│   ├── ur_description/ur3e/       UR3e
│   ├── aubo_description/          遨博 i5
│   ├── inspire_hand/              Inspire 灵巧手重定向配置 yml
│   ├── unitree_hand/、brainco_hand/、g1/、h1_2/
│
├── lib/
│   └── libpyauboi5-v1.5.1...tar.gz 遨博 SDK 的 .so 动态库（用遨博真机时才需要安装）
│
├── requirements.txt               本项目额外依赖
├── 适配真实机械臂.md                ★ 进阶：如何接入一款新的真实机械臂
└── 仿真中机械臂适配md               ★ 进阶：如何给 PyBullet 仿真添加新机械臂
```

---

## 三、环境安装（建议在 Ubuntu 20.04/22.04 下进行）

### 1. 创建 conda 环境

```bash
conda create -n tv python=3.10 -c conda-forge
conda activate tv
# 逆运动学依赖 pinocchio（建议用 conda 装，避免编译问题）
conda install -c conda-forge pinocchio=3.1.0 "numpy<2" casadi
```

### 2. 安装两个本地子模块（在仓库根目录下执行）

```bash
# televuer：XR 数据采集包（证书 cert.pem/key.pem 已在 teleop/televuer/ 目录中，无需再生成）
cd teleop/televuer
pip install -e .
cd ../..

# dex-retargeting：灵巧手重定向算法库（会带上 torch 等较大的依赖）
cd teleop/robot_control/dex-retargeting
pip install -e .
cd ../../..

# 本项目其余依赖（opencv、pybullet、pyserial、pyyaml、meshcat 等）
pip install -r requirements.txt
```

> 如果只跑 PyBullet 仿真、不控制灵巧手，dex-retargeting 也可以暂时不装（但 `teleop_hand_and_arm_sim.py` 间接导入链中包含它，建议一并装上）。

### 3. 按需安装的额外组件

- **遨博 i5 真机**：解压 [lib/libpyauboi5-v1.5.1.x64-for-python3.x.tar.gz](../lib)，把里面的 `.so` 文件复制到当前 Python 环境的 `site-packages` 目录：

  ```bash
  cd lib
  tar -xzf libpyauboi5-v1.5.1.x64-for-python3.x.tar.gz
  cp *.so $(python -c "import site; print(site.getsitepackages()[0])")/
  python -c "from auboi5_robot import Auboi5Robot; print('遨博 SDK 安装成功')"
  ```

- **宇树 G1/H1 真机**：参考官方 README 安装 [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)。

### 3. 硬件准备

| 用途 | 设备 |
|------|------|
| XR 头显 | Meta Quest 3（推荐配手柄）/ Apple Vision Pro / PICO 4 Ultra 企业版 |
| 网络 | 主机与头显连同一局域网（能互相 ping 通） |
| 真机 | 遨博 i5 / UR3e 等机械臂 + Inspire 灵巧手或串口夹爪；主机与机械臂网络互通 |
| 相机（真机） | 头部 USB 相机 + 手腕相机（OpenCV UVC 相机或 Intel RealSense） |
| 仿真 | 一台带 OpenGL 显示的电脑即可，无需机械臂、相机 |

---

## 四、快速上手：三条运行路线

> 所有命令都在**仓库根目录** `xr_teleoperate_robot/` 下执行（因为代码中 URDF、配置均使用相对路径）。

### 路线 A：PyBullet 仿真（新手第一次请走这条，零硬件风险）

```bash
conda activate tv
cd xr_teleoperate_robot          # 进入仓库根目录
python teleop/teleop_hand_and_arm_sim.py
```

启动后会弹出 PyBullet 窗口，加载 Franka Panda 机械臂、桌子和小方块，然后等待头显接入。

### 路线 B：真机单臂 + Inspire 灵巧手

第一步，在**接相机的电脑/工控机**上启动图像服务（真机没有仿真自动画面）：

```bash
# 根据你的相机型号、设备号、分辨率修改 image_server.py 末尾的 config
python teleop/image_server/image_server.py
```

第二步，在**主机**上启动遥操作（默认遨博 i5）：

```bash
python teleop/teleop_hand_and_arm_aubo.py --arm AUBO_I5 --xr-mode controller
# 也可选 UR3E / FRANKA_PANDA：
# python teleop/teleop_hand_and_arm_aubo.py --arm UR3E
```

### 路线 C：真机单臂 + 串口夹爪

```bash
python teleop/teleop_gripper_and_arm.py
```

### 路线 D：宇树 G1/H1 人形双臂（官方原版功能）

```bash
python teleop/teleop_hand_and_arm.py --arm=G1_29 --ee=dex3
```

### 头显连接步骤（所有路线通用）

1. 头显连到与主机**同一 Wi-Fi / 局域网**；
2. 用主机 IP 查一下地址：`ifconfig`（例如 `192.168.123.2`）；
3. 头显内置浏览器打开：

   ```
   https://<主机IP>:8012?ws=wss://<主机IP>:8012
   ```

   出现安全警告属正常（自签名证书），选择「高级 / Advanced → 仍然继续访问」；
4. 点击页面中的 **Virtual Reality** 按钮，进入沉浸模式，就能看到机械臂第一视角；
5. 回到主机终端，按 **`r`** 正式开始遥操作。

### 按键说明

| 按键 | 作用 |
|------|------|
| `r` | 开始遥操作（启动前机械臂不动，用于安全准备） |
| `s` | 开始 / 停止录制一段 episode（需加 `--record`） |
| `h` | 机械臂回到初始姿态（定制单臂脚本中有效） |
| `q` | 退出（退出前机械臂会自动回初始位姿） |

### 常用命令行参数（定制脚本）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--xr-mode` | `controller`（手柄追踪）或 `hand`（裸手手势追踪） | `controller` |
| `--arm` | `AUBO_I5` / `UR3E` / `FRANKA_PANDA` | `AUBO_I5` |
| `--ee` | 末端执行器：`inspire`（灵巧手）/ `inspire_gripper`（夹爪） | — |
| `--frequency` | 主循环频率 Hz | `60.0` |
| `--record` | 启用数据录制，数据保存在 `teleop/utils/data/` | 关闭 |
| `--headless` | 无显示器模式（不弹 OpenCV 窗口） | 关闭 |

---

## 五、跑之前务必检查的几个硬编码配置

定制脚本里有不少写死的 IP / 路径，**换环境后必须按实际情况修改**，否则连不通：

| 文件 | 位置 | 需要改的内容 |
|------|------|--------------|
| [teleop_hand_and_arm_aubo.py](teleop_hand_and_arm_aubo.py) | `ImageClient(server_address=...)` | 图像服务所在电脑的 IP（脚本中为 `192.168.123.116`） |
| 同上 | `create_arm_controller(..., robot_ip=...)` | 遨博 `192.168.123.113` / UR3E `192.168.123.17` |
| [image_server.py](image_server/image_server.py) | 文件末尾 `config` | 相机类型（opencv/realsense）、分辨率、`/dev/video*` 设备号 |
| [image_client.py](image_server/image_client.py) | `__main__` 测试代码中的默认 IP | 图像服务 IP（默认 `192.168.123.164`） |
| [hand_retargeting.py](robot_control/hand_retargeting.py) | `HandType.INSPIRE_HAND` 等 | **绝对路径** `/home/robot/unitree/xr_teleoperate_robot/...`，需改成你本机的仓库路径 |
| [../config/robot_config.yml](../config/robot_config.yml) | 各机器人节 | URDF 路径、关节索引、关节限位、人机臂长缩放比例 |

> 安全提示：真机运行时请站在机械臂工作范围之外，首次测试先用低频率（如 `--frequency 10`），确认动作方向正确再提速。`teleop_hand_and_arm_aubo.py` 主循环内含简单的**碰撞过滤**（末端超出安全包围盒则跳过该帧）。

---

## 六、推荐的读码 / 学习顺序

按这个顺序读，能最快建立整体认识：

1. **本文档** + 根目录 [README_zh-CN.md](../README_zh-CN.md)（了解系统组成与硬件）；
2. [teleop_hand_and_arm_sim.py](teleop_hand_and_arm_sim.py) —— 仿真主程序，看 `while True` 主循环：取数据 → IK → 控制 → 回传图像；
3. [tv_wrapper.py](televuer/src/televuer/tv_wrapper.py) —— 看 `TeleData` 数据结构和 `get_motion_state_data()`，理解头显给了什么数据、坐标系怎么转；
4. [robot_arm_basic.py](robot_control/robot_arm_basic.py) —— 机械臂控制抽象基类：状态订阅线程 + 命令发布线程、限位与速度平滑；
5. [universal_robot_arm.py](robot_control/universal_robot_arm.py) —— UR3E（Socket/URScript）和遨博 i5（SDK）的真实实现；
6. [universal_robot_arm_ik.py](robot_control/universal_robot_arm_ik.py) + [../config/robot_config.yml](../config/robot_config.yml) —— IK 如何由配置驱动，支持任意 6/7 自由度臂；
7. 想加新手或新臂时，再读根目录的 [适配真实机械臂.md](../适配真实机械臂.md) 与 [仿真中机械臂适配md](../仿真中机械臂适配md)。

### 几个关键概念小白卡

- **IK（逆运动学）**：已知「手腕应该到哪个位姿（4×4 矩阵）」，反求「6 个关节各转多少弧度」。本项目用 Pinocchio 建模 + CasADi/IPOPT 做优化求解。
- **重定向（Retargeting）**：把人的 25 个手部关键点，优化映射成灵巧手少数几个关节的角度。
- **共享内存（shared_memory）**：图像线程接收画面后写入一块共享内存，主循环直接读，避免大图反复拷贝。
- **臂长缩放（arm_scaling）**：人手臂和机械臂长度不同，配置文件里用 `human_length / robot_length` 对位移目标做缩放，防止目标超出机械臂工作空间。

---

## 七、常见问题

**Q: 头显浏览器打不开页面 / 一直转圈？**
确认主机防火墙放行 8012 端口、头显与主机在同一网段；URL 里的 IP 要用主机实际 IP；https 证书警告要手动选择继续访问。

**Q: 画面黑屏或没有图像？**
仿真路线的画面由 PyBullet 渲染，必须在有图形界面的机器上运行；真机路线要先启动 `image_server.py`，且 `image_client` 里的 IP、分辨率配置要与服务端一致。

**Q: 提示找不到 `auboi5_robot` / 导入 aubo_sdk 报错？**
没有安装遨博 `.so` 动态库，见上文「环境安装 → 遨博真机」。只用仿真可忽略。

**Q: 提示找不到 Inspire 手的 yml 配置？**
[hand_retargeting.py](robot_control/hand_retargeting.py) 里写死了原作者机器的绝对路径，改成你本机的仓库路径即可。

**Q: 机械臂动作抖动？**
降低 `--frequency`；检查 [robot_config.yml](../config/robot_config.yml) 里的关节限位与臂长缩放；IK 解算结果已经过 `WeightedMovingFilter` 加权平滑，可调整滤波权重。

**Q: 录制的数据在哪？**
默认在 `teleop/utils/data/episode_xxxx/`，包含彩色图像和 states/actions 的 json，可配合宇树 unitree_IL_lerobot 工具转换为训练数据集。
