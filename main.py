import sys
import os
import subprocess
import re
import datetime
import json
import threading
import numpy as np
import scipy.signal
import scipy.io.wavfile
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QComboBox, QPushButton, QMessageBox, QTextEdit)
from PyQt6.QtCore import QProcess, QThread, pyqtSignal
import Audio2TxT
import deepseek

class AudioListener(QThread):
    finished_recording = pyqtSignal(str)
    
    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.is_running = True
        self.process = None
        
        # 音频参数
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16
        self.chunk_duration = 0.05  # 50ms 分片
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # VAD（语音活动检测）参数
        self.vad_threshold = 500  # 根据麦克风灵敏度调整
        self.silence_duration = 1.0  # 触发结束的静音持续时间（秒）
        self.min_speech_duration = 0.5 # 视为有效语音的最小持续时间
        
        self.buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        
    def run(self):
        # ffmpeg 命令，输出原始 PCM 数据到 stdout
        cmd = [
            'ffmpeg', 
            '-f', 'dshow', 
            '-i', f'audio={self.device_name}', 
            '-ar', str(self.sample_rate), 
            '-ac', str(self.channels), 
            '-acodec', 'pcm_s16le', 
            '-f', 's16le', 
            '-'
        ]
        
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            bufsize=self.chunk_size * 2
        )
        
        while self.is_running:
            raw_data = self.process.stdout.read(self.chunk_size * 2) # 每个样本 2 字节
            if not raw_data:
                break
                
            audio_chunk = np.frombuffer(raw_data, dtype=self.dtype)
            
            if len(audio_chunk) == 0:
                continue
                
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
            
            if rms > self.vad_threshold:
                if not self.is_speaking:
                    print("Speech detected...")
                    self.is_speaking = True
                    self.buffer = [] # 开始新的缓冲区
                
                self.silence_counter = 0
                self.buffer.append(audio_chunk)
                
            else:
                if self.is_speaking:
                    self.silence_counter += self.chunk_duration
                    self.buffer.append(audio_chunk)
                    
                    if self.silence_counter >= self.silence_duration:
                        print("Silence detected, processing speech...")
                        self.process_buffer()
                        self.is_speaking = False
                        self.buffer = []
                        self.silence_counter = 0
                        
    def process_buffer(self):
        full_audio = np.concatenate(self.buffer)
        
        # 检查持续时间是否足够长
        duration = len(full_audio) / self.sample_rate
        if duration < self.min_speech_duration:
            print("Speech too short, ignoring.")
            return

        # 1. 带通滤波器 (100Hz - 7500Hz)
        # 300-3400Hz 用于电话，100-7500Hz 更适合宽带 16k 语音
        nyquist = 0.5 * self.sample_rate
        low = 100 / nyquist
        high = 7500 / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        filtered_audio = scipy.signal.lfilter(b, a, full_audio)
        
        # 2. 归一化 (自动增益控制)
        # 提高音量以改善识别
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 0:
            target_peak = 25000 # 目标振幅 (最大 32767)
            # 限制增益以避免过度放大纯噪声 (例如：最大 10 倍增益)
            gain = min(target_peak / max_val, 10.0) 
            if gain > 1.0:
                filtered_audio = filtered_audio * gain
                
        processed_audio = np.clip(filtered_audio, -32768, 32767).astype(np.int16)
        
        # 保存到文件
        save_dir = os.path.join(os.getcwd(), 'records')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"record_{timestamp}.wav"
        filepath = os.path.join(save_dir, filename)
        
        scipy.io.wavfile.write(filepath, self.sample_rate, processed_audio)
        print(f"Saved processed audio to {filepath}")
        
        self.finished_recording.emit(filepath)

    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()

class AudioRecorder(QWidget):
    update_chat_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # 读取配置文件
        self.config = self.load_config()
        self.init_ui()
        # 加载音频设备列表
        self.load_devices()
        
        self.listener = None
        self.is_awake = False
        # 历史问答记录
        self.history = []
        
        # 绑定问答界面更新事件
        self.update_chat_signal.connect(self.append_chat)
        
        # 如果设备可用，自动开始监听
        if self.device_combo.count() > 0:
            self.start_listening()

    def load_config(self):
        config_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def init_ui(self):
        self.setWindowTitle('桌面语音问答助手')
        self.setGeometry(100, 100, 700, 700)

        # 调整全局字体大小（增大 5 个字号）
        font = self.font()
        base_size = font.pointSize()
        if base_size <= 0:
            base_size = 9  # 默认基准
        font.setPointSize(base_size + 3)
        self.setFont(font)

        layout = QVBoxLayout()

        self.label = QLabel("选择麦克风设备:")
        layout.addWidget(self.label)

        self.device_combo = QComboBox()
        layout.addWidget(self.device_combo)

        self.restart_btn = QPushButton("重新启动监听")
        self.restart_btn.clicked.connect(self.start_listening)
        layout.addWidget(self.restart_btn)

        self.status_label = QLabel("正在初始化...")
        layout.addWidget(self.status_label)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        self.setLayout(layout)

    def load_devices(self):
        devices = self.get_audio_devices()
        if not devices:
            self.device_combo.addItem("未找到音频设备")
        else:
            self.device_combo.addItems(devices)

    def get_audio_devices(self):
        devices = []
        try:
            cmd = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
            process = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            output = ""
            try:
                output = process.stderr.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    output = process.stderr.decode('mbcs') 
                except:
                    output = process.stderr.decode('utf-8', errors='ignore')

            lines = output.splitlines()
            audio_section = False
            for line in lines:
                if "DirectShow audio devices" in line:
                    audio_section = True
                    continue
                if "DirectShow video devices" in line:
                    audio_section = False
                    continue
                
                if "Alternative name" in line:
                    continue
                    
                match = re.search(r'\[dshow @ .*?\]\s+"([^"]+)"', line)
                if match:
                    if audio_section or "(audio)" in line:
                        device_name = match.group(1)
                        if device_name not in devices:
                            devices.append(device_name)
        except Exception as e:
            print(f"Error getting devices: {e}")
        return devices

    def start_listening(self):
        if self.listener:
            self.listener.stop()
            self.listener.wait()
            
        device_name = self.device_combo.currentText()
        if not device_name or device_name == "未找到音频设备":
            self.status_label.setText("无效的音频设备")
            return
            
        self.status_label.setText(f"正在监听: {device_name}")
        self.listener = AudioListener(device_name)
        # 绑定录制完毕后的事件
        self.listener.finished_recording.connect(self.on_recording_finished)
        self.listener.start()

    def on_recording_finished(self, filepath):
        self.status_label.setText("正在识别...")
        threading.Thread(target=self.process_audio, args=(filepath,)).start()

    def append_chat(self, text):
        self.chat_display.append(text)
        # 自动滚动
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def process_audio(self, audio_path):
        print(f"Starting recognition for {audio_path}")
        kdxf_config = self.config.get('kdxf', {})
        appid = kdxf_config.get('appid')
        apikey = kdxf_config.get('apikey')
        appsecret = kdxf_config.get('appSecret')

        if not (appid and apikey and appsecret):
            print("Error: Missing KDXF config")
            self.delete_audio(audio_path)
            return

        try:
            text = Audio2TxT.recognize_with_xfyun(audio_path, appid, apikey, appsecret)
            print(f"识别结果: {text}")
            
            # 在识别后立即删除音频
            self.delete_audio(audio_path)
            
            if not text:
                return

            self.status_label.setText("监听中...")

            # 处理结果
            keyTextConfig = self.config.get("keyText", {})
            awake_word = keyTextConfig.get('awake')
            sleep_word = keyTextConfig.get('sleep')

            if awake_word and awake_word in text:
                self.is_awake = True
                self.play_audio(os.path.join(os.getcwd(), 'records', 'here.mp3'))
                self.update_chat_signal.emit(f"User: {text}")
                self.update_chat_signal.emit(f"System: 唤醒成功")
                return
            elif sleep_word and sleep_word in text:
                self.is_awake = False
                self.play_audio(os.path.join(os.getcwd(), 'records', 'exit.mp3'))
                self.update_chat_signal.emit(f"User: {text}")
                self.update_chat_signal.emit(f"System: 进入休眠")
                return

            # DeepSeek 交互
            if self.is_awake:
                self.update_chat_signal.emit(f"User: {text}")
                
                ds_config = self.config.get('deepseek', {})
                api_url = ds_config.get('apiUrl')
                api_key = ds_config.get('apikey')
                system_prompt = ds_config.get('system_prompt', '')
                
                full_answer = ""
                try:
                    self.update_chat_signal.emit("DeepSeek: ") # 新行起始
                    
                    generator = deepseek.ask_stream(system_prompt, text, self.history, api_key, api_url)
                    current_response = ""
                    for chunk in generator:
                        current_response += chunk
                        self.update_chat_signal.emit(f"[STREAM]{chunk}") # 流式传输的特殊前缀
                    
                    self.update_chat_signal.emit("\n") # 消息结束
                    
                    # 更新历史记录
                    self.history.append({'question': text, 'answer': current_response})
                except Exception as e:
                    print(f"\nDeepSeek error: {e}")
                    self.update_chat_signal.emit(f"Error: {e}")
                
        except Exception as e:
            print(f"Recognition error: {e}")
            self.delete_audio(audio_path)

    def delete_audio(self, file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")

    def play_audio(self, file_path):
        if not os.path.exists(file_path):
            return
        try:
            subprocess.run(['ffplay', '-nodisp', '-autoexit', '-hide_banner', file_path], 
                           check=True)
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    # 修改 append_chat 以处理流式传输
    def append_chat(self, text):
        if text.startswith("[STREAM]"):
            chunk = text[8:]
            self.chat_display.moveCursor(self.chat_display.textCursor().MoveOperation.End)
            self.chat_display.insertPlainText(chunk)
        else:
            self.chat_display.append(text)
        
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    ex.show()
    sys.exit(app.exec())
