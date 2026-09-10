import asyncio
import websockets


async def rtde_communication():
    uri = "ws://192.168.123.113:9013"
    async with websockets.connect(uri) as websocket:
        # 发布话题
        publish_topic = '[100,{"chanel":1,"frequency":1.0,"segments":["input_float_registers_0",' \
                        '"input_double_registers_1"],"to_server":true,"trigger":0}] '
        await websocket.send(publish_topic)

        # 向服务器推送数据
        publish_data = '[1,3.3,4.4]'
        await websocket.send(publish_data)

        # 订阅话题
        subscribe_topic = '[100,{"chanel":1,"frequency":50.0,"segments":["input_float_registers_r0",' \
                          '"input_double_registers_r1"],"to_server":false,"trigger":0}] '
        await websocket.send(subscribe_topic)

        # 接收 RTDE 数据包
        while True:
            data_packet = await websocket.recv()
            print("Received data packet:", data_packet)


asyncio.get_event_loop().run_until_complete(rtde_communication())