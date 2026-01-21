# DeskQ  
一款面向工作与学习的桌面智能语音助手，支持**关键词唤醒**，用户**语音提问**，大模型生成的**总结性回答**会自动通过**TTS**（语音合成）回答，实现真正的语音交互体验。依托**LangChain**与本地可配置**RAG知识库**，**说一句话**即可查找、打开并概括文档内容。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/YOUR_GITHUB_NAME/DeskQ)](https://github.com/YOUR_GITHUB_NAME/DeskQ/releases)
[![Python](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/downloads/)

---

<!-- 功能截图 / GIF 预留位 -->
![demo](docs/demo.gif)

## ✨ Features

### 1. 音频采集与处理
- **VAD（语音活动检测）**：基于 RMS 能量检测，通过静态阈值、静音持续时间和有效语音最小持续时间参数精准判断语音起止点。
- **音频质量优化**：引入带通滤波（100-7500Hz）滤除环境噪声，并集成响度归一化与增益控制，大幅提升语音清晰度。

### 2. 语音识别与控制
- **ASR 语音识别**：集成高效语音识别引擎，准确将语音转化为文本指令。
- **关键词唤醒/休眠**：支持自定义语音关键词，实现设备的快速唤醒与低功耗休眠切换。

### 3. 智能问答与知识库
- **LangChain 驱动**：利用 LangChain 框架构建强大的逻辑处理能力。
- **RAG（检索增强生成）**：结合本地知识库，让大模型基于真实文档回答问题，拒绝幻觉。
- **向量数据库支持**：集成 **ElasticSearch** 作为向量检索引擎，实现海量文档的语义检索。
- **多格式支持**：支持 .docx, .doc, .xlsx, .xls, .pdf, .txt, .md 等常见文档格式。

### 4. 语音合成 (TTS)
- **智能语音播报**：自动提取大模型回答的核心总结句，通过 TTS 技术转化为自然流畅的语音并自动播放，实现“听得见”的智能助手。

## 🚀 Start
DeskQ 需要调用外部语音/大模型 API，**首次运行前**请先完成以下配置。

### 前置要求
请确保系统中已安装以下软件并配置好环境变量：
1. **FFmpeg**：用于音频处理与播放（[下载地址](https://ffmpeg.org/download.html)）。
2. **ElasticSearch**：用于向量知识库检索（需启动服务，默认地址 `http://localhost:9200`）。

### 安装步骤

1. 克隆并进入项目
```bash
git clone https://github.com/SM2007421229/DeskQ.git
cd DeskQ
```

2. 填写配置文件
> 把 config.example.json 复制为 config.json，并按下方说明填入你的密钥：
   
| 字段                                 | 说明             | 如何获取/备注                                   |
|:-----------------------------------|:---------------|:------------------------------------------|
| `keyText.awake`                    | 语音唤醒词          | 可自定义，如 `"你好 DeskQ"`                       |
| `keyText.sleep`                    | 语音休眠词          | 可自定义                                      |
| `asr.appid`                        | 科大讯飞开放平台 AppID | [讯飞控制台](https://console.xfyun.cn/) → 创建应用 |
| `asr.apikey`                       | 讯飞 API Key     | 语音听写API Key                               |
| `asr.appSecret`                    | 讯飞 AppSecret   | 语音听写API Secret                            |
| `llm.model`                        | 模型名称           | 如 `deepseek-v3-250324`                    |
| `llm.apiUrl`                       | 大模型接口地址        | 如火山方舟调用地址                                 |
| `llm.apikey`                       | 大模型 API Key    | 对应平台的 API Key                             |
| `llm.system_prompt`                | 人设提示词          | 已内置，可按需调整                                 |
| `vector.host`                      | 向量数据库地址        | 默认为 http://localhost:9200                |
| `vector.user`                      | 向量数据库用户名       | 根据 ES 配置填写                                  |
| `vector.password`                  | 向量数据库密码        | 根据 ES 配置填写                                  |
| `file_knowledge.monitored_folders` | 知识库监控文件夹       | 列表格式，填入本地文件夹绝对路径                          |
| `file_knowledge.file_types`        | 支持的文件后缀        | 列表格式，如 `[".pdf", ".docx", ".xlsx"]`       |
| `file_knowledge.reindex_interval`  | 自动重新索引间隔       | 单位：秒，默认 3600                              |

3. 安装依赖并启动
```bash
pip install -r requirements.txt
python main.py
```

## 📂 Project Structure

```
DeskQ/
├── main.py                 # 主程序入口，PyQt6 界面与业务逻辑
├── records/                # 录音与 TTS 音频缓存目录
├── js/                     # 前端 JavaScript 资源
├── ui_prototype.html       # 聊天界面前端代码
├── config.example.json     # 配置文件示例
├── requirements.txt        # Python 依赖列表
│
│── asr.py                  # 语音识别模块 (XFyun)
│── tts.py                  # 语音合成模块 (XFyun)
│── llm.py                  # 大模型交互模块 (LangChain)
│── file_manager.py         # 文件管理与知识库索引模块
│
└── README.md               # 项目说明文档
```
