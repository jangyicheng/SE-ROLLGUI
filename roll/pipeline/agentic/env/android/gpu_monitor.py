import time
import datetime
import os
import signal
import sys
from pathlib import Path

try:
    import pynvml
except ImportError:
    print("错误: 请先安装 pynvml")
    print("   pip install nvidia-ml-py")
    sys.exit(1)
    
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"成功初始化 NVML，检测到 {device_count} 张 GPU")
except Exception as e:
    print(f"NVML 初始化失败: {e}")
    print("详细错误:")
    traceback.print_exc()
    sys.exit(1)
    
# ================== 配置区 ==================
LOG_DIR = Path("./output/gpu_logs")
LOG_DIR.mkdir(exist_ok=True)

# 日志文件（每天一个文件，或固定一个文件）
LOG_FILE = LOG_DIR / f"gpu_memory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

INTERVAL_SECONDS = 5          # 每隔多少秒记录一次
INCLUDE_UTILIZATION = True    # 是否记录 GPU 计算利用率（%）

# ===========================================

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

print(f"开始监测 {device_count} 张 GPU，每 {INTERVAL_SECONDS} 秒记录一次")
print(f"日志文件: {LOG_FILE.resolve()}")
print("-" * 80)

# 写入表头
with open(LOG_FILE, "a", encoding="utf-8") as f:
    header = "timestamp,gpu_index,gpu_name,total_mb,used_mb,free_mb,used_percent"
    if INCLUDE_UTILIZATION:
        header += ",gpu_util_percent"
    f.write(header + "\n")

def signal_handler(sig, frame):
    print("\n接收到终止信号，正在清理...")
    pynvml.nvmlShutdown()
    print("监测已停止")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    while True:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                # 如果你担心版本回退，可以用 isinstance 判断一下
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                total_mb = mem_info.total // (1024 * 1024)
                used_mb = mem_info.used // (1024 * 1024)
                free_mb = mem_info.free // (1024 * 1024)
                used_percent = round(used_mb / total_mb * 100, 2)
                
                line = f"{timestamp},{i},{name},{total_mb},{used_mb},{free_mb},{used_percent}"
                
                if INCLUDE_UTILIZATION:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    line += f",{gpu_util}"
                
                # 同时打印到控制台和写入文件
                print(line)
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    
            except Exception as e:
                print(f"GPU {i} 获取信息失败: {e}")
        
        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    print(f"发生错误: {e}")
finally:
    pynvml.nvmlShutdown()