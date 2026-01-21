# -*- coding:utf-8 -*-

import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
import subprocess

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        # 在这里通过res_id 来设置通过哪个音库合成
        self.CommonArgs = {"app_id": self.APPID, "status": 2}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {
            "tts": {
                "vcn": "x6_lingyuyan_pro",  # 发音人参数，更换不同的发音人会有不同的音色效果
                "volume": 50,    # 设置音量大小
                "rhy": 0,   # 是否返回拼音标注 0:不返回拼音, 1:返回拼音（纯文本格式，utf8编码）
                "speed": 50,    # 设置合成语速，值越大，语速越快
                "pitch": 50,    # 设置振幅高低，可通过该参数调整效果
                "bgs": 0,   # 背景音 0:无背景音, 1:内置背景音1, 2:内置背景音2
                "reg": 0,   # 英文发音方式 0:自动判断处理，如果不确定将按照英文词语拼写处理（缺省）, 1:所有英文按字母发音, 2:自动判断处理，如果不确定将按照字母朗读
                "rdn": 0,   # 合成音频数字发音方式 0:自动判断, 1:完全数值, 2:完全字符串, 3:字符串优先
                "audio": {
                    "encoding": "lame",  # 合成音频格式， lame 合成音频格式为mp3
                    "sample_rate": 24000,  # 合成音频采样率， 16000, 8000, 24000
                    "channels": 1,  # 音频声道数
                    "bit_depth": 16, # 合成音频位深 ：16, 8
                    "frame_size": 0
                }
            }
        }
        
        self.Data = {
            "text": {
                "encoding": "utf8",
                "compress": "raw",
                "format": "plain",
                "status": 2,
                "seq": 0,
                "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")   # 待合成文本base64格式
            }
        }

class TTSHandler:
    def __init__(self, app_id, api_key, api_secret):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.output_path = os.path.join(os.getcwd(), 'records', 'tts.mp3')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 特定服务的URL
        self.req_url = 'wss://cbm01.cn-huabei-1.xf-yun.com/v1/private/mcd9m97e6'

    def assemble_ws_auth_url(self, requset_url, method="GET"):
        u = self.parse_url(requset_url)
        host = u.host
        path = u.path
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(host, date, method, path)
        
        signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.api_key, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        
        values = {
            "host": host,
            "date": date,
            "authorization": authorization
        }

        return requset_url + "?" + urlencode(values)

    def parse_url(self, requset_url):
        stidx = requset_url.index("://")
        host = requset_url[stidx + 3:]
        schema = requset_url[:stidx + 3]
        edidx = host.index("/")
        if edidx <= 0:
            raise Exception("invalid request url:" + requset_url)
        path = host[edidx:]
        host = host[:edidx]
        
        class Url:
            def __init__(this, host, path, schema):
                this.host = host
                this.path = path
                this.schema = schema
        
        return Url(host, path, schema)

    def on_message(self, ws, message):
        try:
            # print(message)
            message = json.loads(message)
            code = message["header"]["code"]
            sid = message["header"]["sid"]
            
            if "payload" in message:
                audio = message["payload"]["audio"]['audio']
                audio = base64.b64decode(audio)
                status = message["payload"]['audio']["status"]
                
                with open(self.output_path, 'ab') as f:
                    f.write(audio)
                
                if status == 2:
                    print("TTS generation completed.")
                    ws.close()
            
            if code != 0:
                errMsg = message["header"]["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        except Exception as e:
            print("receive msg,but parse exception:", e)

    def on_error(self, ws, error):
        print("### TTS error:", error)

    def on_close(self, ws, *args):
        # print("### TTS closed ###")
        pass

    def on_open(self, ws):
        def run(*args):
            d = {
                "header": self.ws_param.CommonArgs,
                "parameter": self.ws_param.BusinessArgs,
                "payload": self.ws_param.Data,
            }
            d = json.dumps(d)
            # print("------>开始发送文本数据")
            ws.send(d)

        thread.start_new_thread(run, ())

    def generate_audio(self, text):
        if os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
            except Exception as e:
                print(f"Error removing old TTS file: {e}")
                return False

        self.ws_param = Ws_Param(APPID=self.app_id, APIKey=self.api_key, 
                                 APISecret=self.api_secret, Text=text)
        
        websocket.enableTrace(False)
        wsUrl = self.assemble_ws_auth_url(self.req_url)
        ws = websocket.WebSocketApp(wsUrl, 
                                    on_message=self.on_message, 
                                    on_error=self.on_error, 
                                    on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
        return os.path.exists(self.output_path)

    def play_audio(self):
        if not os.path.exists(self.output_path):
            print("No audio file to play.")
            return

        try:
            # 使用 ffplay 播放，-nodisp 不显示窗口，-autoexit 播放完退出，-hide_banner 隐藏版权信息
            cmd = ['ffplay', '-nodisp', '-autoexit', '-hide_banner', self.output_path]
            # 阻塞调用，直到播放结束
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error playing audio: {e}")

    def delete_audio(self):
        if os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
                print("TTS audio file deleted.")
            except Exception as e:
                print(f"Error deleting TTS file: {e}")

def run_tts_task(text, app_id, api_key, api_secret):
    handler = TTSHandler(app_id, api_key, api_secret)
    print(f"Starting Smart TTS for: {text}")
    if handler.generate_audio(text):
        print("Playing Smart TTS audio...")
        handler.play_audio()
        print("Deleting Smart TTS audio...")
        handler.delete_audio()
    else:
        print("Smart TTS generation failed.")

if __name__ == "__main__":
    # 从控制台页面获取以下密钥信息，控制台地址：https://console.xfyun.cn/app/myapp
    appid = 'xxx'
    apisecret = 'xxx'
    apikey = 'xxx'
    
    run_tts_task("全红婵，2007年3月28日出生于广东省湛江市，中国国家跳水队女运动员，主项为女子10米跳台。",
                 appid, apikey, apisecret)
