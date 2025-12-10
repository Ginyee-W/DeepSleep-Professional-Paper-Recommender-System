import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


class ContentEngine:
    def __init__(self,
                 papers_path='papers.csv',
                 embeddings_path='embeddings_fp16.npy',   # 这里用新文件名
                 model_name='all-mpnet-base-v2'):

        # 1. 加载 CSV 数据
        if not os.path.exists(papers_path):
            raise FileNotFoundError(f"未找到论文 CSV 文件: {papers_path}")
        self.df = pd.read_csv(papers_path)

        # 2. 加载 Embeddings 矩阵（磁盘是 float16，内存转回 float32）
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"未找到向量文件: {embeddings_path}")
        emb16 = np.load(embeddings_path)          # float16，小文件
        self.embeddings = emb16.astype('float32')  # 内存里转回 float32，用于 cosine_similarity


        # 3. 加载 Sentence Transformer 模型
        self.model = SentenceTransformer(model_name)

        print("ContentEngine 初始化完成！")

    def _get_paper_info(self, idx, score):
        """辅助函数：格式化单个论文信息，处理 DOI 链接"""
        paper = self.df.iloc[idx]

        # 处理 DOI 链接逻辑
        doi_link = None
        # 尝试读取 'doi' 列，如果 CSV 里叫 'url' 或 'link' 请在这里修改
        raw_doi = paper.get('doi', None)

        if pd.notna(raw_doi):
            raw_doi = str(raw_doi).strip()
            # 如果已经是 http 开头，直接用；否则拼接 doi.org 前缀
            if raw_doi.startswith('http'):
                doi_link = raw_doi
            else:
                doi_link = f"https://doi.org/{raw_doi}"

        return {
            'id': idx,
            'title': paper.get('title', 'No Title'),
            'abstract': paper.get('abstract', 'No Abstract'),
            'doi': doi_link,  # 新增字段
            'score': float(score)
        }

    def search_by_keywords(self, query_text, top_k=5):
        """功能 1: 关键词搜索"""
        query_embedding = self.model.encode([query_text])
        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_indices = similarities[0].argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(self._get_paper_info(idx, similarities[0][idx]))

        return results

    def find_similar_papers(self, paper_index, top_k=5):
        """功能 2: 相似论文推荐"""
        if paper_index >= len(self.embeddings) or paper_index < 0:
            return []

        target_embedding = self.embeddings[paper_index].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, self.embeddings)
        top_indices = similarities[0].argsort()[-(top_k + 1):][::-1]

        results = []
        for idx in top_indices:
            if idx == paper_index:
                continue
            results.append(self._get_paper_info(idx, similarities[0][idx]))

        return results[:top_k]