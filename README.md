# DeskQ  
基于**LangChain**与**RAG**的面向工作与学习的桌面智能语音助手，支持**关键词唤醒**，用户**语音/文字双模式提问**，**说一句话**或**打一行字**即可查找、打开并深度分析 PDF 论文与 Excel 数据报表。大模型生成的**总结性回答**会自动通过**TTS**（语音合成）回答，实现真正的沉浸式交互体验。

[![Python](https://img.shields.io/badge/python-≥3.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/YOUR_GITHUB_NAME/DeskQ)](https://github.com/YOUR_GITHUB_NAME/DeskQ/releases)

---

<!-- 功能截图 / GIF 预留位 -->
![demo](docs/demo.gif)

## ✨ Functions

- **语音/文本双模式问答**：支持沉浸式语音对话与精准文字输入，无缝切换。
- **本地知识库 RAG**：基于本地文件（Excel/PDF）构建知识库，支持深度语义检索与精准回答。
- **智能数据分析**：内置多种统计与计算工具，自动完成 Excel 数据的复杂指标分析。
- **PDF 深度阅读**：保留文档层级结构，支持全文概括与特定章节的深入解析。
- **多轮对话上下文**：具备记忆能力，支持基于历史对话内容的连续追问与上下文理解。
- **语音控制与播报**：支持自定义唤醒词/休眠词，以及大模型回答的自动语音合成播报。

## ✨ Features

### 1. 多模态交互与控制
- **语音/文字双模式**：点击顶部切换按钮即可在沉浸式语音对话和精准文字输入之间无缝切换。
- **音频采集与处理**：基于 RMS 能量的 VAD（语音活动检测），配合 100-7500Hz 带通滤波与响度归一化，精准捕捉每一句指令。
- **语音识别与唤醒**：集成高效 ASR 引擎与自定义关键词唤醒/休眠机制，实现设备的随叫随到与待机。

### 2. 深度 RAG 知识库引擎
- **智能分词**：针对 **Excel** 采用“单行单切片”策略，自动关联表头上下文；针对 **PDF** 采用“层级结构切分”策略，识别章节标题与表格，保留文档逻辑结构。
- **Embedding**：调用Embedding向量模型生成高维语义向量。
- **语义检索**：基于 ElasticSearch 的 Script Score 计算余弦相似度，捕捉深层语义关联。
- **关键词检索**：BM25 算法加权匹配文件名与内容，确保精确术语不丢失。
- **重排序 (Re-ranking)**：按照权重融合向量得分与关键词得分，输出最优候选片段。
- **Excel 数据分析**：自动识别表头与数值列，支持**求和、平均值、百分位数、同比增长率、占比、复合年均增长率 (CAGR)、标准差**工具化计算。
- **PDF 论文问答**：构建多级标题树，支持全文概括与特定章节（如“实验结果”）的深度抽取分析。


### 3. 智能问答与计算引擎
- **LangChain 驱动**：利用 LangChain 框架构建强大的逻辑处理能力，精准识别意图并自动调度下表中的工具。
- **上下文记忆**：内置对话历史管理机制，能够精准理解“它”、“这”等指代词，支持流畅的多轮连续问答。
- **拒绝幻觉**：严格基于检索到的本地文档内容回答，并在回答中注明数据来源。
- **语音合成 (TTS)**：自动提取大模型回答的核心总结句，通过 TTS 技术转化为自然流畅的语音并自动播放。

| 工具方法名称 | 功能描述 |
| :--- | :--- |
| `query_files` | 混合检索（向量+关键词）知识库文件，支持重排序，获取相关片段 |
| `fetch_section_content` | 获取 PDF/文档特定章节（如“实验结果”）的完整层级内容 |
| `open_file` | 调用系统默认程序打开本地文件 |
| `calc_sum_desc` | 计算数值序列的总和（自动处理精度） |
| `calc_mean_desc` | 计算数值序列的算术平均值 |
| `calc_percentile_desc` | 计算数值序列的指定百分位数（如 P95, P99） |
| `calc_growth_rate_desc` | 计算两期数值的同比增长率 |
| `calc_ratio_desc` | 计算两个序列的逐元素比例 |
| `calc_round_desc` | 对数值进行指定位数的四舍五入 |
| `calc_cagr_desc` | 计算复合年均增长率 (CAGR) |
| `calc_std_desc` | 计算样本标准差 |

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
