import sys
import os
import subprocess
import re
import datetime
import json
import threading
import time
import numpy as np
import scipy.signal
import scipy.io.wavfile
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QMessageBox, QHBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import QProcess, QThread, pyqtSignal, QUrl, QObject, pyqtSlot, QFileInfo, Qt, QSize, QEvent
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
import asr
import llm
import file_manager
import tts


# 设置 QtWebEngine 禁用 GPU 加速，解决 0xC0000409 崩溃问题
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["QTWEBENGINE_REMOTE_DEBUGGING_PORT"] = "8333"

class Backend(QObject):
    update_kb_signal = pyqtSignal()
    select_device_signal = pyqtSignal(str)

    @pyqtSlot()
    def trigger_update_kb(self):
        print("Backend: Update KB triggered from JS")
        self.update_kb_signal.emit()

    @pyqtSlot(str)
    def trigger_select_device(self, device_name):
        print(f"Backend: Device selected from JS: {device_name}")
        self.select_device_signal.emit(device_name)

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e293b;
                color: #f1f5f9;
            }
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                padding-left: 10px;
                font-weight: bold;
            }
            QPushButton {
                border: none;
                background-color: transparent;
                color: #94a3b8;
                font-size: 14px;
                width: 32px;
                height: 32px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #ffffff;
            }
            QPushButton#btnClose:hover {
                background-color: #ef4444;
                color: white;
            }
            QPushButton#btnPin:checked {
                color: #818cf8;
                background-color: rgba(129, 140, 248, 0.1);
            }
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title / Icon
        self.title_label = QLabel("🤖 桌面语音问答助手")
        layout.addWidget(self.title_label)
        
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # Buttons
        self.btn_pin = QPushButton("📌")
        self.btn_pin.setObjectName("btnPin")
        self.btn_pin.setToolTip("置顶 (Always on Top)")
        self.btn_pin.setCheckable(True)
        self.btn_pin.clicked.connect(self.toggle_on_top)
        layout.addWidget(self.btn_pin)

        self.btn_min = QPushButton("─")
        self.btn_min.setToolTip("最小化")
        self.btn_min.clicked.connect(self.minimize_window)
        layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("☐")
        self.btn_max.setToolTip("最大化")
        self.btn_max.clicked.connect(self.toggle_maximize)
        layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("btnClose")
        self.btn_close.setToolTip("关闭")
        self.btn_close.clicked.connect(self.close_window)
        layout.addWidget(self.btn_close)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.parent_window:
                self.parent_window.windowHandle().startSystemMove()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()

    def toggle_on_top(self):
        if not self.parent_window:
            return
        
        flags = self.parent_window.windowFlags()
        if self.btn_pin.isChecked():
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        
        self.parent_window.setWindowFlags(flags)
        self.parent_window.show()

    def minimize_window(self):
        if self.parent_window:
            self.parent_window.showMinimized()

    def toggle_maximize(self):
        if not self.parent_window:
            return
            
        if self.parent_window.isMaximized():
            self.parent_window.showNormal()
            self.btn_max.setText("☐")
            self.btn_max.setToolTip("最大化")
        else:
            self.parent_window.showMaximized()
            self.btn_max.setText("❐")
            self.btn_max.setToolTip("向下还原")

    def update_maximize_btn(self, is_maximized):
        if is_maximized:
            self.btn_max.setText("❐")
            self.btn_max.setToolTip("向下还原")
        else:
            self.btn_max.setText("☐")
            self.btn_max.setToolTip("最大化")

    def close_window(self):
        if self.parent_window:
            self.parent_window.close()



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
    update_status_signal = pyqtSignal(str)
    update_awake_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        # 初始化成员变量
        self.listener = None
        self.is_awake = False
        self.current_device_name = None
        self.ai_response_buffer = "" # 缓存AI回答
        self.tts_triggered = False # 是否已触发TTS

        # 读取配置文件
        self.config = self.load_config()

        # 初始化后端桥接
        self.backend = Backend()
        self.backend.update_kb_signal.connect(self.update_knowledge_base)
        self.backend.select_device_signal.connect(self.change_device)

        self.init_ui()

        # 初始化文件管理器
        self.file_manager = file_manager.FileManager(self.config)

        # 初始化 大模型 助手
        llm_config = self.config.get('llm', {})
        self.ai_assistant = llm.chat(
            api_key=llm_config.get('apikey'),
            api_url=llm_config.get('apiUrl'),
            system_prompt=llm_config.get('system_prompt', ''),
            model=llm_config.get('model')
        )

        # 绑定问答界面更新事件
        self.update_chat_signal.connect(self.handle_chat_update)
        self.update_status_signal.connect(self.handle_status_update)
        self.update_awake_signal.connect(self.handle_awake_update)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                self.layout().setContentsMargins(0, 0, 0, 0)
                if hasattr(self, 'title_bar'):
                    self.title_bar.update_maximize_btn(True)
            else:
                self.layout().setContentsMargins(5, 5, 5, 5)
                if hasattr(self, 'title_bar'):
                    self.title_bar.update_maximize_btn(False)
        super().changeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            margin = 5
            rect = self.rect()
            pos = event.position().toPoint()
            edges = Qt.Edge(0)
            
            if pos.x() < margin: edges |= Qt.Edge.LeftEdge
            if pos.x() > rect.width() - margin: edges |= Qt.Edge.RightEdge
            if pos.y() < margin: edges |= Qt.Edge.TopEdge
            if pos.y() > rect.height() - margin: edges |= Qt.Edge.BottomEdge
            
            if edges != Qt.Edge(0):
                if self.windowHandle().startSystemResize(edges):
                    return
                    
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        margin = 5
        rect = self.rect()
        pos = event.position().toPoint()
        
        on_left = pos.x() < margin
        on_right = pos.x() > rect.width() - margin
        on_top = pos.y() < margin
        on_bottom = pos.y() > rect.height() - margin
        
        if on_left and on_top: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif on_right and on_bottom: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif on_left and on_bottom: self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif on_right and on_top: self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif on_left or on_right: self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif on_top or on_bottom: self.setCursor(Qt.CursorShape.SizeVerCursor)
        else: self.setCursor(Qt.CursorShape.ArrowCursor)
        
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

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
        self.setGeometry(100, 100, 800, 800) # Slightly larger for web view
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: #1e293b;")
        self.setMouseTracking(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5) # Minimize margins for full web view look
        layout.setSpacing(0)

        # Custom Title Bar
        self.title_bar = CustomTitleBar(self)
        layout.addWidget(self.title_bar)

        # Web Engine View
        self.web_view = QWebEngineView()
        
        # Setup WebChannel
        self.channel = QWebChannel()
        self.channel.registerObject('backend', self.backend)
        self.web_view.page().setWebChannel(self.channel)

        # Load HTML
        html_path = os.path.abspath("ui_prototype.html")
        self.web_view.loadFinished.connect(self.on_load_finished)
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))

        layout.addWidget(self.web_view)

        self.setLayout(layout)

        self.is_web_loaded = False

    def on_load_finished(self, ok):
        if ok:
            print("Web page loaded successfully.")
            self.is_web_loaded = True

            # Populate devices in HTML
            devices = self.get_audio_devices()
            js_code = f"setDeviceList({json.dumps(devices)});"
            self.web_view.page().runJavaScript(js_code)

            # Start listening on first device if available
            if devices:
                default_device = devices[0]
                self.start_listening(default_device)
                # Sync UI
                self.web_view.page().runJavaScript(f"setCurrentDevice('{default_device}');")
            else:
                self.handle_status_update("未找到音频设备")
        else:
            print("Failed to load web page.")

    def change_device(self, device_name):
        print(f"Switching device to: {device_name}")
        self.start_listening(device_name)

    def handle_status_update(self, text):
        if not hasattr(self, 'is_web_loaded') or not self.is_web_loaded:
            print(f"Web not loaded yet, skipping status: {text}")
            return

        # Call JS setStatus
        js_code = f"setStatus('{text}');"
        self.web_view.page().runJavaScript(js_code)

    def handle_awake_update(self, is_awake):
        if not hasattr(self, 'is_web_loaded') or not self.is_web_loaded:
            return

        status = 'awake' if is_awake else 'sleep'
        js_code = f"setAssistantStatus('{status}');"
        self.web_view.page().runJavaScript(js_code)

    def handle_chat_update(self, text):
        if not hasattr(self, 'is_web_loaded') or not self.is_web_loaded:
            return

        # Dispatch to appropriate JS function
        if text.startswith("[STREAM]"):
            chunk = text[8:]
            
            # 累积文本并检查是否需要触发TTS
            self.ai_response_buffer += chunk
            
            if not self.tts_triggered and '\n' in self.ai_response_buffer:
                first_sentence = self.ai_response_buffer.split('\n')[0].strip()
                if first_sentence:
                    self.tts_triggered = True
                    print(f"Triggering TTS for summary: {first_sentence}")
                    threading.Thread(target=self.run_tts, args=(first_sentence,)).start()
            
            # 转义字符以供 JS 使用
            js_chunk = chunk.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '')
            self.web_view.page().runJavaScript(f"appendAiStream('{js_chunk}');")
            
        elif text.startswith("提问: "):
            content = text[4:].replace('\\', '\\\\').replace("'", "\\'")
            self.web_view.page().runJavaScript(f"addUserMessage('{content}');")
        elif text.startswith("用户: "):
            content = text[4:].replace('\\', '\\\\').replace("'", "\\'")
            self.web_view.page().runJavaScript(f"addUserMessage('{content}');")
        elif text == "回答: ":
            self.ai_response_buffer = "" # 重置缓冲区
            self.tts_triggered = False # 重置TTS触发状态
            self.web_view.page().runJavaScript("startAiMessage();")
        elif text == "\n":
            self.web_view.page().runJavaScript("endAiMessage();")
        elif text.startswith("系统消息: "):
            content = text[6:].replace('\\', '\\\\').replace("'", "\\'")
            # Treat system messages as status or special user messages? 
            # For now, let's just log or set status, or maybe show as a user message for visibility?
            # User wanted "Update KB" to show status.
            self.handle_status_update(content)
        elif text.startswith("Error: "):
            self.handle_status_update(text)

    def run_tts(self, text):
        # 使用配置文件中的 key，如果不存在则使用默认值
        asr_config = self.config.get('asr', {})
        app_id = asr_config.get('appid', '0e0f71f1')
        api_key = asr_config.get('apikey', 'fdeebb4ad3414ce3d9ea703dc2a7369a')
        api_secret = asr_config.get('appSecret', 'M2ZhN2NmOTU5MTk5YTAzNjNlYTY1MDFi')
        
        tts.run_tts_task(text, app_id, api_key, api_secret)

    def update_knowledge_base(self):
        self.update_status_signal.emit("正在更新知识库...")
        threading.Thread(target=self._run_manual_update).start()

    def _run_manual_update(self):
        try:
            if not hasattr(self, 'file_manager'):
                self.file_manager = file_manager.FileManager(self.config)

            self.file_manager.initialize_elasticsearch()
            # 指定要更新的目录，这里假设是配置文件中的第一个目录，或者默认目录
            target_dirs = self.config.get('monitor_paths', [])
            if not target_dirs:
                # Fallback to desktop if not configured
                target_dirs = [os.path.join(os.path.expanduser("~"), "Desktop")]

            print(f"Updating KB for: {target_dirs}")
            self.file_manager.index_files(target_dirs)
            self.update_status_signal.emit("知识库更新完成")
        except Exception as e:
            print(f"Update failed: {e}")
            self.update_status_signal.emit(f"更新失败: {e}")

        # 恢复监听状态显示
        time.sleep(1)
        if self.listener and self.listener.is_running:
            if self.current_device_name:
                self.update_status_signal.emit(f"正在监听: {self.current_device_name}")
        else:
            self.update_status_signal.emit("就绪")

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

    def start_listening(self, device_name=None):
        if self.listener:
            self.listener.stop()
            self.listener.wait()

        if device_name is None:
            if self.current_device_name:
                device_name = self.current_device_name
            else:
                # Should not happen ideally if initialized correctly
                self.handle_status_update("请先选择音频设备")
                return

        self.current_device_name = device_name

        if not device_name or device_name == "未找到音频设备":
            self.handle_status_update("无效的音频设备")
            return

        self.handle_status_update(f"正在监听: {device_name}")
        self.listener = AudioListener(device_name)
        self.listener.finished_recording.connect(self.on_recording_finished)
        self.listener.start()

    def on_recording_finished(self, filepath):
        self.handle_status_update("正在识别...")
        threading.Thread(target=self.process_audio, args=(filepath,)).start()

    def process_audio(self, audio_path):
        print(f"Starting recognition for {audio_path}")
        asr_config = self.config.get('asr', {})
        appid = asr_config.get('appid')
        apikey = asr_config.get('apikey')
        appsecret = asr_config.get('appSecret')

        if not (appid and apikey and appsecret):
            print("语音识别配置错误")
            self.delete_audio(audio_path)
            return

        try:
            text = asr.recognize_with_xfyun(audio_path, appid, apikey, appsecret)
            print(f"识别结果: {text}")

            # 在识别后立即删除音频
            self.delete_audio(audio_path)

            if not text:
                return

            self.update_status_signal.emit("监听中...")

            # 处理结果
            keyTextConfig = self.config.get("keyText", {})
            awake_word = keyTextConfig.get('awake')
            sleep_word = keyTextConfig.get('sleep')

            if awake_word and awake_word in text:
                self.is_awake = True
                self.update_awake_signal.emit(True)
                self.play_audio(os.path.join(os.getcwd(), 'records', 'here.mp3'))
                self.update_chat_signal.emit(f"用户: {text}")
                self.update_chat_signal.emit(f"系统消息: 唤醒成功\n")
                return
            elif sleep_word and sleep_word in text:
                self.is_awake = False
                self.update_awake_signal.emit(False)
                self.play_audio(os.path.join(os.getcwd(), 'records', 'exit.mp3'))
                self.update_chat_signal.emit(f"用户: {text}")
                self.update_chat_signal.emit(f"系统消息: 进入休眠")
                return

            # 大模型 交互
            if self.is_awake:
                self.update_chat_signal.emit(f"提问: {text}")

                try:
                    self.update_chat_signal.emit("回答: ")  # 新行起始

                    # 使用 self.ai_assistant 进行流式对话，session_id 固定为 "user_main_session"
                    generator = self.ai_assistant.ask_stream(text, session_id="user_main_session")

                    for chunk in generator:
                        self.update_chat_signal.emit(f"[STREAM]{chunk}")  # 流式传输的特殊前缀

                    self.update_chat_signal.emit("\n")  # 消息结束

                except Exception as e:
                    print(f"\n小助手发生错误: {e}")
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    ex.show()
    sys.exit(app.exec())
