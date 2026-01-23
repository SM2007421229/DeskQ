# DeskQ  
一款面向工作与学习的桌面智能语音助手，支持**关键词唤醒**，用户**语音/文字双模式提问**，大模型生成的**总结性回答**会自动通过**TTS**（语音合成）回答，实现真正的沉浸式交互体验。依托**LangChain**与本地可配置**RAG知识库**，**说一句话**或**打一行字**即可查找、打开并深度分析 PDF 论文与 Excel 数据报表。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/YOUR_GITHUB_NAME/DeskQ)](https://github.com/YOUR_GITHUB_NAME/DeskQ/releases)
[![Python](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/downloads/)

---

<!-- 功能截图 / GIF 预留位 -->
![demo](docs/demo.gif)

## ✨ Features

### 1. 多模态交互与控制
- **语音/文字双模式**：点击顶部切换按钮即可在沉浸式语音对话和精准文字输入之间无缝切换。
- **音频采集与处理**：基于 RMS 能量的 VAD（语音活动检测），配合 100-7500Hz 带通滤波与响度归一化，精准捕捉每一句指令。
- **语音识别与唤醒**：集成高效 ASR 引擎与自定义关键词唤醒/休眠机制，实现设备的随叫随到与待机。

### 2. 深度 RAG 知识库引擎
- **智能分词与 Embedding**：使用 `RecursiveCharacterTextSplitter` 对文本进行语义分块，并调用向量模型生成高维语义向量。
- **多路召回 (Hybrid Retrieval)**：
  - **语义检索**：基于 ElasticSearch 的 Script Score 计算余弦相似度，捕捉深层语义关联。
  - **关键词检索**：BM25 算法加权匹配文件名与内容，确保精确术语不丢失。
  - **重排序 (Re-ranking)**：按照权重融合向量得分与关键词得分，输出最优候选片段。
- **多格式深度解析**：
  - **Excel 数据分析**：自动识别表头与数值列，支持**求和、平均值、百分位数、同比增长率、占比、复合年均增长率 (CAGR)、标准差**工具化计算。
  - **PDF 论文问答**：构建多级标题树，支持全文概括与特定章节（如“实验结果”）的深度抽取分析。

### 3. 智能问答与工具链
- **LangChain 驱动**：利用 LangChain 框架构建强大的逻辑处理能力，无缝调度计算工具与检索工具。
- **拒绝幻觉**：严格基于检索到的本地文档内容回答，并在回答中注明数据来源。
- **语音合成 (TTS)**：自动提取大模型回答的核心总结句，通过 TTS 技术转化为自然流畅的语音并自动播放。

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
| `vector.apiUrl`                    | 向量模型接口地址      | 用于生成文本 Embedding                       |
| `vector.apikey`                    | 向量模型 API Key    | 对应平台的 API Key                          |
| `es.host`                          | ES 地址           | 默认为 http://localhost:9200                |
| `es.user`                          | ES 用户名          | 根据 ES 配置填写                          |
| `es.password`                      | ES 密码           | 根据 ES 配置填写                          |
| `file_knowledge.monitored_folders` | 知识库监控文件夹       | 列表格式，填入本地文件夹绝对路径                          |
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
├── tool.py                 # LangChain 工具集 (计算工具、混合检索、文件打开)
├── file_manager.py         # 核心 RAG 引擎 (文件解析、Embedding、ES 索引与检索)
├── smart-tts.py            # 智能 TTS 模块 (可选独立运行)
├── records/                # 录音与 TTS 音频缓存目录
├── js/                     # 前端 JavaScript 资源 (qwebchannel, marked, mathjax)
├── ui_prototype.html       # 聊天界面前端代码 
├── config.example.json     # 配置文件示例
├── requirements.txt        # Python 依赖列表
│
│── asr.py                  # 语音识别模块 (XFyun)
│── tts.py                  # 语音合成模块 (XFyun)
│── llm.py                  # 大模型交互模块 (LangChain ChatModel)
│
└── README.md               # 项目说明文档
```
