import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import time
import random 
from streamlit_echarts import st_echarts 

from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Module Import & Error Handling
# ==========================================

try:
    from graph_engine import GraphRecommender
except ImportError:
    GraphRecommender = None

try:
    import agent_module
except ImportError:
    agent_module = None


# ==========================================
# 1. Resource Loading (Fixed Title Issue)
# ==========================================

@st.cache_resource
def load_resources():
    res = {}
    try:
        # A. Load Data
        df = pd.read_csv('papers.csv')
        df['work_id'] = df['work_id'].astype(str)
        if 'doi' not in df.columns: df['doi'] = ''

        # --- DATA CLEANING (CRITICAL FIX) ---

        # 1. Clean Title (Fix for 'float object is not subscriptable')
        if 'title' not in df.columns:
            df['title'] = 'Untitled'
        else:
            # Fill NaN with 'Untitled' and force convert to string
            df['title'] = df['title'].fillna('Untitled').astype(str)

        # 2. Clean Abstract
        fallback_msg = "Please visit the homepage to view."
        if 'abstract' not in df.columns:
            df['abstract'] = fallback_msg
        else:
            df['abstract'] = df['abstract'].astype(str).apply(
                lambda x: fallback_msg if x.lower() == 'nan' or not x.strip() else x
            )

        # 3. Clean Metadata columns
        cols_to_check = ['publication_date', 'authors', 'institution_countries', 'cited_by_count']
        for c in cols_to_check:
            if c not in df.columns:
                df[c] = 'N/A'
            else:
                df[c] = df[c].fillna('N/A')

        res['df'] = df

        # B. Load Vectors
        if os.path.exists('embeddings.npy'):
            vectors = np.load('embeddings.npy').astype('float32')
            res['vectors'] = vectors
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(vectors)
            index.add(vectors)
            res['index'] = index

        # C. Load Model
        res['model'] = SentenceTransformer('all-mpnet-base-v2')

        # D. Load Graph Engine
        if GraphRecommender and os.path.exists("paper_graph.gpickle"):
            res['graph_engine'] = GraphRecommender(graph_path="paper_graph.gpickle")
        else:
            res['graph_engine'] = None

        return res
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None


# ==========================================
# 2. Core Recommendation Logic
# ==========================================

def get_personalized_recommendation(user_id, query, resources, alpha=0.5, beta=0.5, top_k=5):
    """
    Revised logic with Dynamic Hybrid Fusion.
    """
    # Initialize results dictionary
    results = {'content': pd.DataFrame(), 'graph': pd.DataFrame(), 'fusion': pd.DataFrame()}
    
    df = resources['df']
    index = resources.get('index')
    model = resources['model']
    vectors = resources.get('vectors')
    graph_engine = resources.get('graph_engine')

    history_ids = st.session_state.get(f"history_{user_id}", [])
    current_graph_anchor = st.session_state.get('graph_anchor_id', None)

    # Helper: Filter out read papers
    def filter_read_and_get_ids(indices):
        return [i for i in indices if i not in history_ids and i < len(df)]

    # -------------------------------------------------
    # A. Content-based Retrieval
    # -------------------------------------------------
    content_candidates = {} # {idx: score (0-1)}
    
    # Use search query if available, otherwise use history vector
    search_vec = None
    if query and index:
        search_vec = model.encode([query]).astype('float32')
    elif history_ids and vectors is not None:
        valid_hist = [i for i in history_ids if i < len(vectors)]
        if valid_hist:
            target_vecs = vectors[valid_hist]
            search_vec = np.mean(target_vecs, axis=0).reshape(1, -1).astype('float32')
            
    if search_vec is not None and index:
        faiss.normalize_L2(search_vec)
        # Retrieve 3x top_k to give fusion algorithm enough candidates
        D, I = index.search(search_vec, top_k * 3)
        
        # Store for display
        raw_idxs = filter_read_and_get_ids(I[0])
        results['content'] = df.iloc[raw_idxs[:top_k]].copy()
        
        # Normalize scores (0-1)
        for rank, idx in enumerate(raw_idxs):
            if idx < len(D[0]):
                score = float(D[0][rank]) 
                content_candidates[idx] = score

        # Anchor logic: If no specific anchor, use top search result
        if current_graph_anchor is None and len(raw_idxs) > 0:
            current_graph_anchor = df.iloc[raw_idxs[0]]['work_id']

    # -------------------------------------------------
    # B. Graph-based Retrieval
    # -------------------------------------------------
    graph_candidates = {} # {idx: score (normalized)}
    
    if graph_engine and current_graph_anchor:
        pairs = graph_engine.get_hybrid_recommendation(str(current_graph_anchor), top_k=top_k * 3)
        if pairs:
            # Map work_ids back to DataFrame indices
            cand_wids = [p[0] for p in pairs]
            mask = df['work_id'].isin(cand_wids)
            found_df = df[mask]
            
            # Normalize scores based on max occurrences
            max_score = pairs[0][1] if pairs[0][1] > 0 else 1 
            wid_score_map = {p[0]: (p[1] / max_score) for p in pairs}
            
            # Store for display (filter read)
            clean_graph_df = found_df[~found_df.index.isin(history_ids)]
            results['graph'] = clean_graph_df.head(top_k).copy()
            
            # Record scores
            for idx, row in clean_graph_df.iterrows():
                wid = row['work_id']
                graph_candidates[idx] = wid_score_map.get(wid, 0.0)

    # -------------------------------------------------
    # C. [CORE] Dynamic Hybrid Fusion
    # -------------------------------------------------
    
    # 1. Union of candidates
    all_candidate_idxs = set(content_candidates.keys()) | set(graph_candidates.keys())
    
    fusion_scores = []
    for idx in all_candidate_idxs:
        s_content = content_candidates.get(idx, 0.0)
        s_graph = graph_candidates.get(idx, 0.0)
        
        # === Weighted Sum ===
        final_score = (alpha * s_content) + (beta * s_graph)
        
        fusion_scores.append((idx, final_score))
    
    # 2. Sort by final score
    fusion_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Extract Top K
    final_top_idxs = [x[0] for x in fusion_scores[:top_k]]
    
    if final_top_idxs:
        results['fusion'] = df.iloc[final_top_idxs].copy()
        
    return results, current_graph_anchor


# ==========================================
# 3. AI Agent Logic
# ==========================================

def ai_agent_summary(text):
    """
    Calls agent_module and returns the raw dictionary for flexible UI rendering.
    """
    if not text or len(str(text)) < 10:
        return {"error": "⚠️ Input text is too short to summarize."}

    if not agent_module:
        return {"error": "⚠️ `agent_module.py` not found."}

    with st.spinner('🤖 AI Agent is reading and analyzing (Powered by LLM)...'):
        try:
            # agent_module.generate_summary returns a dict: {'summary':..., 'innovation':..., 'results':...}
            return agent_module.generate_summary(text)
        except Exception as e:
            return {"error": f"⚠️ Generation Failed: {str(e)}"}

# ==========================================
# 4. Graph Generation Helper (New Feature)
# ==========================================

def generate_enhanced_graph_for_main(center_row, rec_df, full_df):
    """
    Graph Generator for main_app (Visual Enhanced - English)
    """
    nodes = []
    links = []

    # 1. Center Node
    if center_row is None or center_row.empty:
        return [], []
    
    if isinstance(center_row, pd.DataFrame):
        center_row = center_row.iloc[0]

    center_id = str(center_row['work_id'])
    center_title = str(center_row['title'])
    short_center = (center_title[:15] + '...') if len(center_title) > 15 else center_title

    nodes.append({
        "id": "CENTER", "name": "CENTER", "symbolSize": 60, "value": 100, "category": 0,
        "label": {"show": True, "formatter": short_center, "fontSize": 14, "fontWeight": "bold", "color": "#FFFFFF"},
        "itemStyle": {"color": "#FF4B4B", "shadowBlur": 20, "shadowColor": "rgba(255, 75, 75, 0.5)"},
        "tooltip": {"formatter": f"📍 <b>Current Context</b><br>{center_title}"} # <--- English
    })

    # 2. Recommended Nodes
    fake_score = 90
    for _, row in rec_df.iterrows():
        rid = str(row['work_id'])
        title = str(row['title'])
        short_title = (title[:12] + '..') if len(title) > 12 else title
        
        nodes.append({
            "id": rid, "name": rid, "symbolSize": 30, "category": 1, "value": fake_score,
            "label": {"show": True, "formatter": short_title, "fontSize": 11, "color": "#A6E1FA"},
            "itemStyle": {"color": "#00C0F2", "shadowBlur": 10, "shadowColor": "rgba(0, 192, 242, 0.4)"},
            "tooltip": {"formatter": f"🔗 <b>Recommendation</b><br>{title}"} # <--- English
        })
        
        links.append({
            "source": "CENTER", "target": rid,
            "lineStyle": {"width": 3, "curveness": 0.1, "color": "rgba(200, 200, 200, 0.3)"}
        })

        # 3. Visual Enhancement (Background Nodes)
        num_satellites = random.randint(2, 4)
        for j in range(num_satellites):
            sat_id = f"{rid}_sub_{j}"
            nodes.append({
                "id": sat_id, "name": sat_id, "symbolSize": random.randint(5, 10), "category": 2, "value": 10,
                "label": {"show": False},
                "itemStyle": {"color": "#606060", "opacity": 0.6},
                "tooltip": {"formatter": "📄 <b>Potential Citation</b><br>Secondary Reference"} # <--- English
            })
            links.append({
                "source": rid, "target": sat_id,
                "lineStyle": {"width": 1, "curveness": 0.2, "color": "rgba(100, 100, 100, 0.2)"}
            })
            if j > 0 and random.random() > 0.6:
                 links.append({"source": sat_id, "target": f"{rid}_sub_{j-1}", "lineStyle": {"width": 0.5, "curveness": 0, "color": "rgba(100, 100, 100, 0.1)"}})
        
        fake_score -= 5

    return nodes, links

# ==========================================
# 5. Report Generation Helper (New Feature)
# ==========================================
from datetime import datetime

def generate_markdown_report(df, history_indices, ai_insights):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"# 📑 AI Research Report\n"
    md += f"**Generated Date:** {date_str}\n"
    md += f"**Papers Reviewed:** {len(history_indices)}\n\n"
    md += "---\n\n"

    for idx in reversed(history_indices):
        if idx >= len(df): continue
        
        row = df.iloc[idx]
        wid = str(row['work_id'])
        title = str(row['title'])
        
        md += f"## 📄 {title}\n"
        md += f"- **ID:** `{wid}`\n"
        md += f"- **Authors:** {str(row.get('authors', 'N/A'))}\n"
        md += f"- **Year:** {str(row.get('publication_date', 'N/A'))}\n\n"
        
        if wid in ai_insights:
            insight = ai_insights[wid]
            md += "### 🤖 AI Agent Analysis\n"
            md += f"**📘 Core Summary:**\n> {insight.get('summary', 'N/A')}\n\n"
            md += f"**💡 Innovation:**\n> {insight.get('innovation', 'N/A')}\n\n"
            md += f"**📊 Key Results:**\n> {insight.get('results', 'N/A')}\n\n"
        else:
            md += "### 📝 Abstract\n"
            abstract = str(row.get('abstract', ''))
            md += f"{abstract[:500]}..." if len(abstract) > 500 else abstract
            md += "\n\n"
            
        md += "---\n\n"
    
    md += "\n*Generated by Pro Paper Recommender System*"
    return md

# ==========================================
# 6. Frontend UI
# ==========================================

def main():
    st.set_page_config(page_title="Pro Paper Recommender", layout="wide")

    # --- State Initialization ---
    if 'page' not in st.session_state: st.session_state.page = 'home'
    if 'selected_paper' not in st.session_state: st.session_state.selected_paper = None
    if 'user_id' not in st.session_state: st.session_state.user_id = 'user_001'
    if 'search_query' not in st.session_state: st.session_state.search_query = ''
    if 'graph_anchor_id' not in st.session_state: st.session_state.graph_anchor_id = None
    if 'ai_insights' not in st.session_state: st.session_state.ai_insights = {}

    hist_key = f"history_{st.session_state.user_id}"
    if hist_key not in st.session_state: st.session_state[hist_key] = []

    res = load_resources()
    if not res: st.stop()
    df = res['df']

    # --- Sidebar: Reading History ---
    with st.sidebar:
        st.header(f"📖 Reading History ({len(st.session_state[hist_key])})")
        if st.button("🗑️ Clear All History", type="primary"):
            st.session_state[hist_key] = []
            st.rerun()
        st.divider()
        current_history = list(st.session_state[hist_key])
        if not current_history: st.caption("No papers read yet.")
        for i, idx in enumerate(reversed(current_history)):
            if idx < len(df):
                paper_row = df.iloc[idx]
                # Safely handle title display here too
                display_title = str(paper_row['title'])

                c_text, c_btn = st.columns([5, 1])
                c_text.markdown(f"**{i + 1}. {display_title[:40]}...**")
                if c_btn.button("✖️", key=f"del_hist_{idx}"):
                    st.session_state[hist_key].remove(idx)
                    st.rerun()
                st.markdown("---")

        if st.session_state[hist_key]: 
            report_content = generate_markdown_report(
                df, 
                st.session_state[hist_key], 
                st.session_state.ai_insights
            )
            
            st.download_button(
                label="📥 Export Research Report (MD)",
                data=report_content,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                type="primary"  # 醒目的样式
            )
        
        st.divider()

        st.header("🎛️ Fusion Control")
        st.info("Adjust weights to change the hybrid recommendation logic.")
        
        # Sliders for Alpha and Beta
        alpha = st.slider("🧠 Content Weight", 0.0, 1.0, 0.5, 0.1)
        beta = st.slider("🕸️ Graph Weight", 0.0, 1.0, 0.5, 0.1)
        
        st.caption(f"Strategy: {int(alpha*100)}% Content + {int(beta*100)}% Graph")
        st.divider()

    # --- Page 1: Home Dashboard ---
    if st.session_state.page == 'home':
        st.title("📚 Professional Paper Recommender")

        # Search Callback to reset Graph Anchor
        def on_search_change():
            st.session_state.graph_anchor_id = None

        query = st.text_input(
            "Search Papers:",
            value=st.session_state.search_query,
            placeholder="e.g. Deep Learning, Transformer...",
            on_change=on_search_change
        )
        st.session_state.search_query = query
        st.markdown("")

        recs, anchor_used = get_personalized_recommendation(
            st.session_state.user_id, 
            query, 
            res,
            alpha=alpha,  
            beta=beta      
        )

        def render_card(row, type_):
            is_active_anchor = (str(row['work_id']) == str(st.session_state.graph_anchor_id))
            
            border_color = "red" if is_active_anchor else None 
            
            with st.container(border=True):
                # Safely handle title display
                title_str = str(row['title'])
                
                if is_active_anchor and type_ == 'cnt':
                    st.markdown(f"**🔴 {title_str[:60]}...**")
                else:
                    st.markdown(f"**{title_str[:60]}...**")
                
                st.caption(f"ID: {row['work_id']}")

                b1, b2 = st.columns([1, 1.5])
                
                with b1:
                    def on_click_details(r):
                        st.session_state.selected_paper = r
                        st.session_state.page = 'details'
                        st.session_state.graph_anchor_id = r['work_id']

                    st.button(
                        "📄 Details",
                        key=f"{type_}_det_{row['work_id']}",
                        on_click=on_click_details,
                        args=(row,)
                    )
                
                if type_ == 'cnt':
                    with b2:
                        def on_click_visualize(rid):
                            st.session_state.graph_anchor_id = rid
                            
                        if is_active_anchor:
                            st.button("👁️ Viewing", key=f"viz_{row['work_id']}", disabled=True)
                        else:
                            st.button(
                                "🕸️ Visualize", 
                                key=f"viz_{row['work_id']}",
                                on_click=on_click_visualize,
                                args=(row['work_id'],) 
                            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.header("🔍 Search Match")
            if recs['content'].empty:
                st.info("No matches or all matches read.")
            else:
                for _, r in recs['content'].iterrows(): render_card(r, 'cnt')

        with c2:
            st.header("🕸️ Graph Context")
            
            center_row = None
            if st.session_state.graph_anchor_id:
                center_res = df[df['work_id'] == str(st.session_state.graph_anchor_id)]
                if not center_res.empty:
                    center_row = center_res.iloc[0]
                    st.info(f"Based on: **{str(center_row['title'])[:40]}...**")
                else:
                    st.warning("Selected paper not found in database.")

            if recs['graph'].empty:
                st.info("No citation connections found.")
            else:
                view_mode = st.radio(
                    "View Mode", 
                    ["List View", "Interactive Graph"], 
                    horizontal=True, 
                    label_visibility="collapsed",
                    key="view_mode_toggle"
                )
                
                st.markdown("---") 

                if view_mode == "List View":
                    for _, r in recs['graph'].iterrows(): 
                        render_card(r, 'gph')

                else:
                    if center_row is not None:
                        nodes, links = generate_enhanced_graph_for_main(center_row, recs['graph'], df)
                        
                        options = {
                            "backgroundColor": "transparent",
                            "tooltip": {"trigger": "item"},
                            "legend": {"show": False},
                            "series": [{
                                "type": "graph", "layout": "force",
                                "data": nodes, "links": links,
                                "categories": [{"name": "CENTER"}, {"name": "Recommended"}, {"name": "Context"}],
                                "roam": True, "draggable": True,
                                "label": {"position": "right"},
                                "force": {"repulsion": 250, "gravity": 0.1, "edgeLength": [30, 80]},
                                "lineStyle": {"curveness": 0.3}
                            }]
                        }
                        
                        st_echarts(
                            options=options, 
                            height="400px", 
                            key=f"echarts_{st.session_state.graph_anchor_id}"
                        )
                    else:
                        st.caption("Select a paper to see the graph.")

        with c3:
            st.header("❤️ For You")
            st.caption(f"Hybrid Score (α={alpha}, β={beta})")
            
            if recs['fusion'].empty:
                st.info("No recommendations yet. Try searching or clicking a paper.")
            else:
                for _, r in recs['fusion'].iterrows(): 
                    render_card(r, 'fus')

    # --- Page 2: Details View ---
    elif st.session_state.page == 'details':
        p = st.session_state.selected_paper
        fallback_msg = "Please visit the homepage to view."

        if st.button("⬅️ Back to Dashboard"):
            st.session_state.page = 'home'
            st.rerun()

        # Safe title display
        st.title(str(p['title']))

        # Metadata Section
        st.info(f"""
        **📅 Publication Date:** {p.get('publication_date', 'N/A')}  
        **👥 Authors:** {p.get('authors', 'N/A')}  
        **🏛️ Institution Countries:** {p.get('institution_countries', 'N/A')}  
        **📉 Cited By:** {p.get('cited_by_count', 'N/A')}
        """)

        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.subheader("Abstract")
            sys_abs = p['abstract']
            st.write(sys_abs)
            st.markdown("---")

            st.subheader("🧠 AI Agent Analysis")
            user_input = st.text_area("Paste abstract here if missing above:", height=150,
                                      placeholder="Paste text here...")

            # Check if we already have the insight in memory (Cache hit!)
            current_wid = str(p['work_id'])
            
            # 如果 session 里已经有了，直接用，不用再花钱调 API
            if current_wid in st.session_state.ai_insights:
                res_dict = st.session_state.ai_insights[current_wid]
                st.success("Loaded from memory cache! ⚡")
                # 直接渲染
                st.divider()
                st.markdown("### 📘 Core Summary")
                st.info(res_dict.get('summary', 'N/A'))
                st.markdown("### 💡 Key Innovations")
                st.success(res_dict.get('innovation', 'N/A'))
                st.markdown("### 📊 Main Results")
                st.warning(res_dict.get('results', 'N/A'))

            else:
                # 只有当 button 点击 且 内存里没有时，才生成
                if st.button("✨ Generate Real AI Summary"):
                    text_to_use = user_input if user_input.strip() else (sys_abs if sys_abs != fallback_msg else "")
                    
                    if text_to_use:
                        res_dict = ai_agent_summary(text_to_use) # 调用 Agent
                        
                        if "error" in res_dict:
                            st.error(res_dict["error"])
                        else:
                            # [关键修改] 保存结果到 Session State
                            st.session_state.ai_insights[current_wid] = res_dict # <--- 存下来！
                            
                            st.success("Analysis Complete & Saved!")
                            st.divider()
                            
                            st.markdown("### 📘 Core Summary")
                            st.info(res_dict.get('summary', 'N/A'))
                            st.markdown("### 💡 Key Innovations")
                            st.success(res_dict.get('innovation', 'N/A'))
                            st.markdown("### 📊 Main Results")
                            st.warning(res_dict.get('results', 'N/A'))

        with c_right:
            st.info("Actions")
            idx = p.name
            if idx in st.session_state[hist_key]:
                st.button("✅ Already Read", disabled=True)
            else:
                if st.button("📖 Mark as Read"):
                    st.session_state[hist_key].append(idx)
                    st.toast("Added to reading history!")
                    time.sleep(0.5)
                    st.rerun()
            if pd.notna(p.get('doi')) and str(p.get('doi')).strip():
                link = p['doi'] if str(p['doi']).startswith('http') else f"https://doi.org/{p['doi']}"
                st.link_button("🌐 Open Source Text", link)


if __name__ == "__main__":
    main()