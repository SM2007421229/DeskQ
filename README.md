# DeskQ  
一款面向工作与学习的桌面智能语音助手，支持**关键词唤醒**和自然对话，依托**LangChain**与本地可配置**RAG知识库**，**说一句话**即可查找、打开并概括文档内容

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/YOUR_GITHUB_NAME/DeskQ)](https://github.com/YOUR_GITHUB_NAME/DeskQ/releases)
[![Python](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/downloads/)

---

&lt;!-- 功能截图 / GIF 预留位 --&gt;
![demo](docs/demo.gif)

## ✨ Features
- **基于 RMS 能量检测的 VAD（语音活动检测）**：通过静态阈值、静音持续时间和有效语音最小持续时间参数判断有效语音起止点。
- **优化音频质量**：引入 带通滤波（100-7500Hz） 以滤除环境噪声，并集成响度归一化与增益控制，提升语音清晰度。
- **唤醒/休眠机制**：语音关键词实现唤醒/休眠机制。
- **轻量级向量检索 RAG**：内置自研 SimpleVectorDB（基于 HashingVectorizer），无需外部向量数据库依赖，即可实现本地文件名的语义模糊匹配与高效索引。
- **多格式文档支持**：支持读取 .docx, .doc, .xlsx, .xls, .pdf, .txt, .md 等多种格式文件内容。
- **智能 Agent 模式**：大模型可自主决定调用工具查询文件列表或读取文件内容，实现精准的文档问答。

## 🚀 Start
DeskQ 需要调用外部语音/大模型 API，**首次运行前**请先完成以下两步配置。

1. 克隆并进入项目
```bash
git clone https://github.com/SM2007421229/DeskQ.git
cd DeskQ
```
2. 填写配置文件
> 把 config.example.json 复制为 config.json，并按下方说明填入你的密钥：
   
| 字段 | 说明 | 如何获取/备注 |
| :--- | :--- | :--- |
| `keyText.awake` | 语音唤醒词 | 可自定义，如 `"你好 DeskQ"` |
| `keyText.sleep` | 语音休眠词 | 可自定义 |
| `asr.appid` | 科大讯飞开放平台 AppID | [讯飞控制台](https://console.xfyun.cn/) → 创建应用 |
| `asr.apikey` | 讯飞 API Key | 语音听写API Key |
| `asr.appSecret` | 讯飞 AppSecret | 语音听写API Secret |
| `llm.model` | 模型名称 | 如 `deepseek-v3-250324` |
| `llm.apiUrl` | 大模型接口地址 | 如火山方舟调用地址 |
| `llm.apikey` | 大模型 API Key | 对应平台的 API Key |
| `llm.system_prompt` | 人设提示词 | 已内置，可按需调整 |
| `file_knowledge.monitored_folders` | 知识库监控文件夹 | 列表格式，填入本地文件夹绝对路径 |
| `file_knowledge.file_types` | 支持的文件后缀 | 列表格式，如 `[".pdf", ".docx", ".xlsx"]` |
| `file_knowledge.reindex_interval` | 自动重新索引间隔 | 单位：秒，默认 3600 |

3. 安装依赖并启动
```bash
pip install -r requirements.txt
python main.py
```
