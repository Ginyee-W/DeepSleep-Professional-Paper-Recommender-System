Professional Paper Recommender System (PPRS)
基于混合图算法与 AI Agent 的科研论文智能推荐助手

Hybrid Graph Recommendation | Interactive Knowledge Galaxy | AI Research Report Generation

📖 项目简介 (Introduction)
本项目是一个面向科研人员的智能论文推荐系统。区别于传统的关键词匹配，本系统采用 “内容语义 + 引用拓扑” 的双路混合推荐策略。
它不仅能通过 Embedding 向量检索内容相似的论文，还能基于引文耦合（Bibliographic Coupling）和共被引（Co-citation）算法发现隐性的学术关联。同时，系统内置了 AI Agent，能够自动阅读摘要并生成结构化的创新点总结，并支持一键导出 Markdown 研报。

✨ 核心功能 (Key Features)
1. 混合检索 (Hybrid Search)
语义检索：基于 SentenceTransformer 和 Faiss 向量库，理解查询意图。
图谱推荐：融合了文献耦合和共被引两种策略，推荐更精准。

2. 交互式知识星云 (Interactive Knowledge Galaxy)
使用 ECharts 渲染高性能力导向图。
具备视觉增强 (Visual Enhancement) 技术，自动生成模拟的二级引用背景场，展示学术网络结构。

3. AI 智能研报 (AI Agent Analysis)
自动提取论文的 Core Summary（核心摘要）、Key Innovation（关键创新） 和 Main Results（主要结果）。
支持记忆功能（Session Cache），避免重复消耗 Token。

4. 研报一键导出 (Report Generation)
自动追踪用户的阅读历史。
将看过的论文元数据与 AI 笔记整合，一键生成排版精美的 .md 研究报告。

📂 项目结构 (File Structure)
1. 核心程序
main_app.py：[主程序] Streamlit 前端入口。集成了搜索、图谱交互、AI 总结及研报导出功能。
graph_engine.py：[图算法引擎] 后端推荐核心。
实现了 Bibliographic Coupling (相似度) 和 Co-citation (经典度) 的混合加权算法。
包含图谱构建与 .gpickle 存取逻辑。
agent_module.py：[AI Agent 模块] 调用 LLM (如 OpenAI/DeepSeek) 对论文摘要进行深度总结。
content_engine.py：(可选) 负责基于内容的 Faiss 向量检索逻辑。

2. 数据文件
papers.csv：原始论文元数据（包含 Title, Abstract, Authors, Year 等）。
embeddings.npy / embeddings.faiss：预训练好的论文向量索引文件，用于加速语义搜索。
paper_graph.gpickle：预计算好的 NetworkX 图结构文件（直接加载，无需每次重新构建）。
reference.txt：原始引用关系数据（用于构建图谱）。

3. 开发环境
requirements.txt：项目依赖库清单（包含 streamlit, pandas, networkx, streamlit-echarts 等）。
app_A/B/C.py：早期的独立功能测试模块（Legacy）。

🚀 快速运行 (Quick Start)
1. 环境准备
确保已安装 Python 3.8+，并安装所需依赖：
pip install -r requirements.txt

2. 启动系统
在项目根目录下运行以下命令：
streamlit run main_app.py --server.fileWatcherType none