import serial
import time
import threading

# 打开串口：端口名根据实际情况修改，如 /dev/ttyTHS1
ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=0)  # 设置timeout为0，非阻塞模式

# 确认串口是否打开
if ser.isOpen():
    print("串口已打开，开始发送和接收数据")

# 接收数据的函数
def receive_data():
    while True:
        if ser.in_waiting > 0:  # 检查是否有数据可读取
            received_data = ser.read(ser.in_waiting).decode('utf-8').strip()  # 读取所有可用数据
            print(f"接收到的数据: {received_data}")

# 启动接收数据的线程
receive_thread = threading.Thread(target=receive_data, daemon=True)
receive_thread.start()

# 发送数据的函数
def send_data():
    ser.write(b'666\n')  # 发送“666”
    print("已发送：666")
    # 每1秒再次调用send_data函数
    threading.Timer(1, send_data).start()

# 启动定时发送数据
send_data()

# 程序保持运行状态
try:
    while True:
        time.sleep(1)  # 让主线程保持活跃

except KeyboardInterrupt:
    print("手动终止程序")

finally:
    ser.close()
    print("串口已关闭")
