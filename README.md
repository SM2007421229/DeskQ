# DeskQ  
桌面语音问答助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/YOUR_GITHUB_NAME/DeskQ)](https://github.com/YOUR_GITHUB_NAME/DeskQ/releases)
[![Python](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/downloads/)

---

&lt;!-- 功能截图 / GIF 预留位 --&gt;
![demo](docs/demo.gif)

## ✨ Features
- 🎤 离线/在线语音识别（支持 Vosk & Azure）
- 🧠 自定义问答知识库（JSON / Markdown）
- 🔌 插件式 TTS 引擎（微软、Edge、系统 TTS）
- 🪶 轻量低占用，启动 &lt; 200 MB
- 🌍 完全开源，MIT 协议

## 🚀 Quick Start
DeskQ 需要调用外部语音/大模型 API，**首次运行前**请先完成以下两步配置。

1. 克隆并进入项目
```bash
git clone https://github.com/YOUR_GITHUB_NAME/DeskQ.git
cd DeskQ
```
2. 填写配置文件
> 把 config.example.json 复制为 config.json，并按下方说明填入你的密钥：
   
| 字段                       | 说明                 | 如何获取                                              |
| :------------------------: | :------------------: | :-------------------------------------------------: |
| `keyText.awake`          | 语音唤醒词，默认 `"小助手"`   | 可自定义，如 `"你好 DeskQ"`                               |
| `keyText.sleep`          | 语音休眠词，默认 `"退下吧"`   | 可自定义                                              |
| `kdxf.appid`             | 科大讯飞开放平台 AppID     | [讯飞控制台](https://console.xfyun.cn/) → 创建应用 |
| `kdxf.apikey`            | 讯飞 API Key         | 语音听写API Key                                                |
| `kdxf.appSecret`         | 讯飞 AppSecret       | 语音听写API Secret                                                |
| `deepseek.apiUrl`        | 火山方舟 DeepSeek 接口地址 | 方舟控制台 → 模型部署页 → 调用地址                              |
| `deepseek.apikey`        | 火山方舟 API Key       | 方舟控制台 → 开通大模型后创建API Key                                      |
| `deepseek.system_prompt` | 大模型人设提示词           | 已内置，可按需调整                                         |

3. 安装依赖并启动
```bash
pip install -r requirements.txt
python main.py
```
