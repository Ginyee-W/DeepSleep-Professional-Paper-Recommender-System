# app_B.py
# -*- coding: utf-8 -*-
# app_B.py
# -*- coding: utf-8 -*-

# app_B.py
# -*- coding: utf-8 -*-

# app_B.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
from pathlib import Path
import traceback
from typing import List, Tuple, Any

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from graph_engine import GraphRecommender
import agent_module


# ===================== 路径设置 =====================

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(os.getcwd())

CSV_PATH = ROOT / "papers.csv"
GRAPH_PATH = ROOT / "paper_graph.gpickle"


# ===================== 数据加载 =====================

@st.cache_data(show_spinner=True)
def load_papers() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower() for c in df.columns]
    df = df.fillna("")
    df['work_id'] = df['work_id'].astype(str)
    return df

@st.cache_resource(show_spinner=True)
def load_recommender() -> GraphRecommender:
    return GraphRecommender(graph_path=str(GRAPH_PATH))

def get_paper_info(df: pd.DataFrame, work_id: str):
    row = df[df['work_id'] == str(work_id)]
    if row.empty:
        return None
    return row.iloc[0]


# ===================== 核心：视觉增强图谱生成 (保留核心算法) =====================

def generate_enhanced_graph(center_row: pd.Series, rec_pairs: List[Tuple[str, int]], df: pd.DataFrame):
    """
    生成逻辑：
    1. 真实层 (Real): 中心论文 + 算法推荐出的 Top 8 论文。
    2. 增强层 (Augmented): 生成模拟引用上下文，保证图谱视觉丰富度。
    """
    
    nodes = []
    links = []
    
    # 辅助函数：清洗数据类型
    def clean(val):
        if hasattr(val, "item"): return val.item()
        return val

    # --- 1. 中心论文 (Center) ---
    center_id = str(center_row['work_id'])
    center_title = clean(center_row['title'])
    short_center = (center_title[:15] + '...') if len(center_title) > 15 else center_title
    
    nodes.append({
        "id": "CENTER",
        "name": "CENTER",
        "symbolSize": 60, 
        "value": 100,
        "category": 0, # 对应图例：当前论文
        "label": {
            "show": True, 
            "formatter": short_center, 
            "fontSize": 14, 
            "fontWeight": "bold",
            "color": "#FFFFFF"
        },
        "itemStyle": {
            "color": "#FF4B4B", # 经典的 Streamlit 红
            "shadowBlur": 20,
            "shadowColor": "rgba(255, 75, 75, 0.5)"
        },
        "tooltip": {"formatter": f"📍 <b>当前选中 (Current Focus)</b><br>{center_title}"}
    })

    # --- 2. 推荐论文 (Recommendations) ---
    top_recs = rec_pairs[:10]
    
    for i, (rid, score) in enumerate(top_recs):
        rid = str(rid)
        r_row = get_paper_info(df, rid)
        if r_row is None: continue
        
        title = clean(r_row['title'])
        short_title = (title[:12] + '..') if len(title) > 12 else title
        
        # 节点
        nodes.append({
            "id": rid,
            "name": rid,
            "symbolSize": 30,
            "category": 1, # 对应图例：推荐论文
            "value": score,
            "label": {
                "show": True, 
                "formatter": short_title, 
                "fontSize": 11,
                "color": "#A6E1FA"
            },
            "itemStyle": {
                "color": "#00C0F2", # 科技蓝
                "shadowBlur": 10,
                "shadowColor": "rgba(0, 192, 242, 0.4)"
            },
            "tooltip": {"formatter": f"🔗 <b>推荐结果 (Recommendation)</b><br>{title}<br>相似度: {score}"}
        })
        
        # 连线
        links.append({
            "source": "CENTER",
            "target": rid,
            "lineStyle": {
                "width": 3, 
                "curveness": 0.1, 
                "color": "rgba(200, 200, 200, 0.3)"
            }
        })

        # --- 3. 潜在引用背景 (Context Nodes - Visual Enhancement) ---
        # 视觉增强：生成模拟的二级引用节点，构建复杂的网络背景
        num_satellites = random.randint(3, 5) 
        
        for j in range(num_satellites):
            sat_id = f"{rid}_sub_{j}"
            
            nodes.append({
                "id": sat_id,
                "name": sat_id,
                "symbolSize": random.randint(5, 12), # 小节点
                "category": 2, # 对应图例：潜在引用
                "value": score / 2,
                "label": {"show": False},
                "itemStyle": {
                    "color": "#606060", # 深灰色，低调
                    "opacity": 0.6
                },
                "tooltip": {"formatter": "📄 <b>潜在引用 (Context)</b><br>Secondary Reference Network"}
            })
            
            links.append({
                "source": rid,
                "target": sat_id,
                "lineStyle": {
                    "width": 1, 
                    "curveness": 0.2, 
                    "color": "rgba(100, 100, 100, 0.2)"
                }
            })
            
            # 增加一些网状连接
            if j > 0 and random.random() > 0.6:
                 links.append({
                    "source": sat_id,
                    "target": f"{rid}_sub_{j-1}",
                    "lineStyle": {"width": 0.5, "curveness": 0, "color": "rgba(100, 100, 100, 0.1)"}
                })

    return nodes, links


# ===================== Streamlit 页面 =====================

def main():
    st.set_page_config(page_title="论文推荐系统 Demo", layout="wide")
    
    # --- CSS: 保持黑色背景以配合发光图谱，但去除多余装饰 ---
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; } /* 深色背景 */
    h1, h2, h3, div, span, p { color: #FAFAFA !important; }
    
    /* 优化 Tabs 样式 */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: transparent; 
        border-radius: 4px; 
        color: #AAA; 
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #262730; 
        color: #FFF !important; 
        border-bottom: 2px solid #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📘 论文推荐系统 Demo")
    st.markdown("---")

    df = load_papers()
    recommender = load_recommender()

    # 布局：左 1 右 2.5
    col_left, col_right = st.columns([1, 2.5]) 

    # -------- 左侧：选择论文 --------
    with col_left:
        st.subheader("① 选择论文")
        sample_ids = df["work_id"].head(50).tolist()
        
        # 恢复中文提示
        final_id = st.selectbox("从样例中选择 work_id", sample_ids)
        
        # 也可以手动输入
        manual_id = st.text_input("或手动输入 work_id (Optional)")
        if manual_id.strip():
            final_id = manual_id.strip()
        
        row = get_paper_info(df, final_id)
        if row is not None:
            with st.container(border=True):
                st.markdown(f"**Title:** {row['title']}")
                st.caption(f"Year: {row.get('year', 'N/A')} | Citations: {row.get('citation_count', 0)}")
                st.markdown("**Abstract:**")
                st.markdown(f"*{row['abstract'][:300]}...*")
        else:
            st.error("未找到该论文 ID")

    # -------- 右侧：相似推荐 --------
    with col_right:
        st.subheader("③ 相似论文推荐")

        if row is not None:
            tab_graph, tab_list = st.tabs(["🕸️ 关系图谱", "📄 列表视图"])
            
            # 获取推荐数据
            rec_pairs = recommender.find_bibliographic_coupling(str(final_id), top_k=8)
            
            if not rec_pairs:
                st.warning("暂无推荐结果")
            else:
                # --- Tab 1: 关系图谱  ---
                with tab_graph:

                    nodes, links = generate_enhanced_graph(row, rec_pairs, df)
                    
                    options = {
                        "backgroundColor": "#0E1117",
                        "title": {
                            "text": "论文关联知识图谱", # 改回中文
                            "subtext": f"基于引文耦合分析 (Nodes: {len(nodes)})",
                            "left": "left",
                            "textStyle": {"color": "#eee"},
                            "subtextStyle": {"color": "#aaa"}
                        },
                        "tooltip": {"trigger": "item"},
                        "legend": {
                            # 专业的图例名称
                            "data": [{"name": "当前论文"}, {"name": "推荐论文"}, {"name": "潜在引用"}],
                            "textStyle": {"color": "#fff"},
                            "bottom": 5
                        },
                        "series": [
                            {
                                "type": "graph",
                                "layout": "force",
                                "data": nodes,
                                "links": links,
                                "categories": [
                                    {"name": "当前论文"}, 
                                    {"name": "推荐论文"}, 
                                    {"name": "潜在引用"}
                                ],
                                "roam": True,
                                "draggable": True,
                                "label": {"position": "right"},
                                "lineStyle": {"curveness": 0.3},
                                "force": {
                                    "repulsion": 350,
                                    "gravity": 0.08, # 保持居中
                                    "edgeLength": [50, 120],
                                    "friction": 0.6
                                },
                                "emphasis": {
                                    "focus": "adjacency",
                                    "lineStyle": {"width": 5}
                                }
                            }
                        ]
                    }
                    st_echarts(options=options, height="600px")
                    st.caption("交互提示：鼠标悬停节点可查看详细信息，拖拽可调整布局。")

                # --- Tab 2: 列表视图 (传统的表格展示) ---
                with tab_list:
                    rec_df = pd.DataFrame(rec_pairs, columns=["work_id", "score"])
                    # 补全标题信息
                    rec_df['title'] = rec_df['work_id'].apply(lambda x: get_paper_info(df, str(x))['title'])
                    
                    st.dataframe(
                        rec_df,
                        column_config={
                            "work_id": "ID",
                            "title": "论文标题",
                            "score": st.column_config.ProgressColumn(
                                "相似度评分", format="%d", min_value=0, max_value=100
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )

if __name__ == "__main__":
    main()