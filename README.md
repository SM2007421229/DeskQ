# DeskQ  
面向工作与学习场景的桌面智能语音助手，支持**关键词语音唤醒**与自然对话，能够实时理解用户意图并调用AI大模型进行精准、流畅的交互，提供高效、便捷的智能问答体验。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/YOUR_GITHUB_NAME/DeskQ)](https://github.com/YOUR_GITHUB_NAME/DeskQ/releases)
[![Python](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/downloads/)

---

&lt;!-- 功能截图 / GIF 预留位 --&gt;
![demo](docs/demo.gif)

## ✨ Features
- 基于 RMS 能量检测的 VAD（语音活动检测），通过静态阈值、静音持续时间和有效语音最小持续时间参数判断有效语音起止点。
- 针对音频质量进行优化，引入 带通滤波（100-7500Hz） 以滤除环境噪声，并集成响度归一化与增益控制，提升语音清晰度。
- 语音关键词实现唤醒/休眠机制。
- 每次提问会将历史问答记录作为上下文提交至大模型，实现连续对话理解。

## 🚀 Start
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
| `keyText.awake`          | 语音唤醒词   | 可自定义，如 `"你好 DeskQ"`                               |
| `keyText.sleep`          | 语音休眠词   | 可自定义                                              |
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
