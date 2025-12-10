import streamlit as st
import pandas as pd
from content_engine import ContentEngine
import re

# Set page configuration
st.set_page_config(page_title="Smart Paper Search Demo", layout="wide")

# --- State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'search'
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ''
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None

# === Feature Add: Read Status Tracking ===
if 'read_papers' not in st.session_state:
    st.session_state.read_papers = set()


# --- Resource Loading ---
@st.cache_resource
def load_engine():
    try:
        return ContentEngine(
            papers_path='papers.csv',
            embeddings_path='embeddings.npy',
            model_name='all-mpnet-base-v2'
        )
    except FileNotFoundError as e:
        st.error(f"Startup Failed: {e}")
        return None


# --- Translation Helper ---
def translate_query_if_needed(query_text):
    if re.search(r'[\u4e00-\u9fff]', query_text):
        try:
            from deep_translator import GoogleTranslator
            with st.spinner(f"Translating '{query_text}' to English..."):
                translator = GoogleTranslator(source='auto', target='en')
                translated_text = translator.translate(query_text)
            st.info(f"🔄 Auto-translated: **{query_text}** ➡️ **{translated_text}**")
            return translated_text
        except ImportError:
            st.warning("⚠️ Chinese input detected, but `deep-translator` is not installed.")
            st.code("pip install deep-translator", language="bash")
            return query_text
        except Exception as e:
            st.warning(f"Translation failed: {e}. Searching with original text.")
            return query_text
    return query_text


# --- Page Navigation & Logic Helpers ---
def go_to_details(paper_info):
    st.session_state.selected_paper = paper_info
    st.session_state.page = 'details'


def back_to_search():
    st.session_state.page = 'search'


def toggle_read_status(paper_id):
    """Toggle the read status of a paper."""
    if paper_id in st.session_state.read_papers:
        st.session_state.read_papers.remove(paper_id)
    else:
        st.session_state.read_papers.add(paper_id)


# --- Main Application ---
def main():
    engine = load_engine()
    if engine is None:
        return

    # >>>>>>>>> Page 1: Search Results <<<<<<<<<
    if st.session_state.page == 'search':
        st.title("📚 Smart Paper Search")

        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter keywords (Chinese or English):", value=st.session_state.last_query,
                                  placeholder="e.g. 深度学习, machine learning")
        with col2:
            st.write("")
            st.write("")
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_btn or (query and query != st.session_state.last_query):
            if query.strip():
                effective_query = translate_query_if_needed(query)
                with st.spinner('Calculating vector similarity...'):
                    st.session_state.search_results = engine.search_by_keywords(effective_query, top_k=10)
                    st.session_state.last_query = query
            else:
                st.warning("Please enter search keywords.")

        if st.session_state.search_results:
            st.markdown(f"### Found {len(st.session_state.search_results)} relevant papers")

            # Show progress of how many you've read in this search result
            read_count = sum(1 for res in st.session_state.search_results if res['id'] in st.session_state.read_papers)
            if read_count > 0:
                st.caption(f"You have read {read_count} papers from this list.")

            st.markdown("---")

            for res in st.session_state.search_results:
                is_read = res['id'] in st.session_state.read_papers

                # Visual Indicator for Read papers
                title_prefix = "✅ [READ] " if is_read else ""
                bg_color = "background-color: #f0f2f6;" if is_read else ""  # Optional styling hook

                with st.container():
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.subheader(f"{title_prefix}{res['title']}")
                        abstract = res['abstract'] if isinstance(res['abstract'], str) else "No abstract available"
                        st.write(f"{abstract[:200]}...")
                        st.caption(f"ID: {res['id']} | Similarity: {res['score']:.4f}")

                    with c2:
                        st.write("")
                        # === Mark as Read Button ===
                        btn_label = "❌ Unmark" if is_read else "✅ Mark Read"
                        st.button(btn_label, key=f"read_toggle_{res['id']}", on_click=toggle_read_status,
                                  args=(res['id'],))

                        # === Find Similar Button ===
                        st.button("🔗 Find Similar",
                                  key=f"btn_{res['id']}",
                                  on_click=go_to_details,
                                  args=(res,),
                                  help="Find other papers similar to this one")

                        # === DOI Link ===
                        if res.get('doi'):
                            st.link_button("🌐 Full Text", res['doi'])

                    st.markdown("---")

        elif st.session_state.last_query:
            st.info("No matching results found.")

    # >>>>>>>>> Page 2: Details & Recommendations <<<<<<<<<
    elif st.session_state.page == 'details':
        paper = st.session_state.selected_paper

        st.button("⬅️ Back to Search", on_click=back_to_search)
        st.markdown("---")

        # Details Header
        c_title, c_action = st.columns([5, 2])
        is_read = paper['id'] in st.session_state.read_papers

        with c_title:
            title_prefix = "✅ " if is_read else ""
            st.title(f"{title_prefix}{paper['title']}")

        with c_action:
            st.write("")
            c_act1, c_act2 = st.columns(2)
            with c_act1:
                # === Toggle Read Status in Details ===
                btn_label = "❌ Unmark" if is_read else "✅ Mark Read"
                # Use a specific key for details page to avoid conflict
                st.button(btn_label, key=f"detail_read_{paper['id']}", on_click=toggle_read_status, args=(paper['id'],))
            with c_act2:
                if paper.get('doi'):
                    st.link_button("🌐 Go to DOI", paper['doi'], type="primary")

        st.caption(f"Paper ID: {paper['id']} | Original Search Similarity: {paper['score']:.4f}")

        full_abstract = paper['abstract'] if isinstance(paper['abstract'], str) else "No abstract available"
        st.info(f"**Abstract:**\n\n{full_abstract}")

        st.markdown("### 🔥 You Might Also Like (Similar Papers)")

        with st.spinner("Analyzing semantic similarity..."):
            recommendations = engine.find_similar_papers(paper['id'], top_k=4)

        if recommendations:
            row1 = st.columns(2)
            row2 = st.columns(2)
            grid_cols = row1 + row2

            for i, sim_paper in enumerate(recommendations):
                sim_is_read = sim_paper['id'] in st.session_state.read_papers

                with grid_cols[i]:
                    with st.container(border=True):
                        # Add a small checkmark in title if read
                        sim_prefix = "✅ " if sim_is_read else ""
                        st.markdown(f"#### {sim_prefix}{sim_paper['title']}")

                        # 确保分数不超过 1.0
                        safe_score = min(float(sim_paper['score']), 1.0)
                        # 确保分数不小于 0.0 (保险起见)
                        safe_score = max(safe_score, 0.0)

                        st.progress(safe_score, text=f"Similarity: {sim_paper['score']:.4f}")

                        sim_abs = sim_paper['abstract'] if isinstance(sim_paper['abstract'],
                                                                      str) else "No abstract available"
                        st.caption(f"{sim_abs[:100]}...")

                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            st.button("Find Similar",
                                      key=f"rec_{sim_paper['id']}",
                                      on_click=go_to_details,
                                      args=(sim_paper,))
                        with col_btn2:
                            # Also allow marking read directly from recommendations?
                            # Maybe keep it simple for now, just show status.
                            if sim_paper.get('doi'):
                                st.link_button("🌐 Full Text", sim_paper['doi'])

        else:
            st.warning("Not enough similar papers found.")


if __name__ == "__main__":
    main()