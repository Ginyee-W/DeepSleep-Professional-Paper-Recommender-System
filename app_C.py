import streamlit as st
import pandas as pd
import numpy as np
import faiss


# ==========================================
# 1. 核心后端逻辑
# ==========================================

@st.cache_resource
def load_data():
    print("正在初始化数据 (只运行一次)...")
    try:
        # 加载 CSV
        df = pd.read_csv('papers.csv')
        # 简单容错：如果没有 doi 列，就造一个假的
        if 'doi' not in df.columns:
            df['doi'] = 'https://google.com'

        # 加载向量
        vectors = np.load('embeddings.npy')
        index = faiss.read_index('embeddings.faiss')

        return df, vectors, index
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None, None


def get_recommendations(history_ids, vectors, index, df, top_k=5):
    """
    根据历史 ID 返回推荐结果 DataFrame
    """
    if not history_ids:
        # 如果没有历史，默认返回前 top_k 个
        # 修复：切片要用 [:top_k]
        return df.iloc[:top_k].copy()

    # 1. 取出历史向量并计算平均值
    history_vecs = vectors[history_ids]
    user_vector = np.mean(history_vecs, axis=0).reshape(1, -1).astype('float32')

    # 2. 搜索
    D, I = index.search(user_vector, top_k + len(history_ids))

    # 3. 整理结果
    rec_indices = []
    for idx in I[0]:
        if idx not in history_ids:
            rec_indices.append(idx)
            # 修复：判断是否达到数量用 >=
            if len(rec_indices) >= top_k:
                break

    # 返回对应的 DataFrame 行
    return df.iloc[rec_indices].copy()


# ==========================================
# 2. 前端网页逻辑
# ==========================================

def main():
    # 修复：加上引号
    st.set_page_config(page_title="论文推荐系统", layout="wide")

    st.title("📚 智能论文推荐系统")
    st.caption("基于向量检索与用户画像技术的实时推荐演示")

    # --- Step 1 加载数据 ---
    df, vectors, index = load_data()
    if df is None:
        return  # 数据没加载成功就停止

    # --- Step 2 初始化用户记忆 (Session State) ---
    if 'history' not in st.session_state:
        st.session_state.history = []  # 初始化为空列表

    # --- Step 3 侧边栏 - 显示用户画像 ---
    with st.sidebar:
        st.header("👤 用户画像")
        # 修复：加上引号和f-string格式
        st.write(f"已阅读文章数: {len(st.session_state.history)}")

        if st.session_state.history:
            st.write("最近阅读记录:")
            # 修复：[-3:] 表示取最后3个
            recent_ids = st.session_state.history[-3:]
            for rid in recent_ids:
                title = df.iloc[rid]['title']
                # 修复：[:20] 表示取前20个字
                st.text(f"- {title[:20]}...")

        # 重置按钮
        if st.button("🗑️ 清空历史 (重置画像)"):
            st.session_state.history = []
            st.rerun()  # 立即刷新页面

    # --- Step 4 主界面 - 推荐展示 ---

    rec_df = get_recommendations(st.session_state.history, vectors, index, df)

    st.subheader("🎯 为您精选的论文")

    # 遍历推荐结果
    for i, row in rec_df.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                # 修复：加上加粗 markdown 和引号
                st.markdown(f"**{row['title']}**")

                # 处理 DOI 链接
                doi_link = row['doi']
                if not str(doi_link).startswith('http'):
                    # 修复：补全链接格式
                    doi_link = f"https://doi.org/{doi_link}"

                st.markdown(f"[🔗 点击查看原文]({doi_link})")

            with col2:
                # 定义点击后的动作
                def on_click_read(paper_id):
                    st.session_state.history.append(paper_id)

                # 渲染按钮
                st.button(
                    "📖 我读过了",
                    key=f"btn_{i}",
                    on_click=on_click_read,
                    args=(i,)
                )

    # --- 调试信息 ---
    st.divider()
    with st.expander("查看当前算法状态 (Debug)"):
        st.write("当前历史 ID 列表:", st.session_state.history)


if __name__ == "__main__":
    main()