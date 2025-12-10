import pandas as pd
import networkx as nx
import pickle
import os

import pandas as pd
import networkx as nx
import pickle
import os

class GraphRecommender:
    def __init__(self, references_path=None, graph_path=None):
        """
        初始化推荐器。
        """
        self.G = None
        
        # 1. 加载现成的图
        if graph_path and os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}...")
            try:
                with open(graph_path, 'rb') as f:
                    self.G = pickle.load(f)
            except Exception as e:
                print(f"Error loading graph: {e}")
        
        # 2. 从原始数据构建
        elif references_path and os.path.exists(references_path):
            print(f"Building graph from {references_path}...")
            self.build_graph_from_data(references_path)
        
        else:
            print("Warning: No data loaded. Please provide a valid path.")

    def build_graph_from_data(self, data_path):
        """
        读取数据并构建有向图。
        """
        # 注意：根据你的数据实际情况调整 sep (逗号 ',' 或 制表符 '\t')
        try:
            df = pd.read_csv(data_path, sep='\t')
        except:
            df = pd.read_csv(data_path) # Fallback to comma
            
        print("Data preview:")
        print(df.head())
        
        # 构建有向图：Paper_From -> Paper_To (引用关系)
        self.G = nx.from_pandas_edgelist(
            df, 
            source='paper_id_from', 
            target='paper_id_to', 
            create_using=nx.DiGraph()
        )
        print(f"✅ Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

    def save_graph(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"💾 Graph saved to {output_path}")

    # ====================================================
    # 算法 A: 文献耦合 (找同行 - 读了同样书的人)
    # ====================================================
    def find_bibliographic_coupling(self, paper_id, top_k=20):
        if self.G is None or paper_id not in self.G: return []

        # 1. 我引用了谁？ (Out-degree)
        my_refs = set(self.G.successors(paper_id))
        if not my_refs: return []

        scores = {}
        for ref in my_refs:
            # 2. 谁也引用了这些人？
            peers = self.G.predecessors(ref)
            for peer in peers:
                if peer == paper_id: continue
                scores[peer] = scores.get(peer, 0) + 1
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ====================================================
    # 算法 B: 共被引 (找经典 - 被同一群人引用的人)
    # ====================================================
    def find_co_citation(self, paper_id, top_k=20):
        if self.G is None or paper_id not in self.G: return []

        # 1. 谁引用了我？ (In-degree)
        cited_by_who = set(self.G.predecessors(paper_id))
        if not cited_by_who: return []

        scores = {}
        for parent in cited_by_who:
            # 2. 这些人还引用了谁？
            siblings = self.G.successors(parent)
            for sibling in siblings:
                if sibling == paper_id: continue
                scores[sibling] = scores.get(sibling, 0) + 1
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ====================================================
    # 🏆 算法 C: 混合推荐策略 (Hybrid)
    # ====================================================
    def get_hybrid_recommendation(self, paper_id, top_k=6, weights=(0.6, 0.4)):
        """
        融合 文献耦合(BC) 和 共被引(CC) 的结果。
        weights: (bc_weight, cc_weight) 默认更看重文献耦合(内容相似)
        """
        # 1. 获取两组候选列表 (取更多候选以供融合)
        bc_list = self.find_bibliographic_coupling(paper_id, top_k=50)
        cc_list = self.find_co_citation(paper_id, top_k=50)
        
        if not bc_list and not cc_list:
            return []

        # 2. 归一化分数的简单处理 (转成字典)
        # 这种简单的加权相加对于 Demo 足够了
        final_scores = {}
        
        # 处理文献耦合 (BC)
        if bc_list:
            max_bc = bc_list[0][1] # 最高分
            for pid, score in bc_list:
                norm_score = score / max_bc # 归一化到 0-1
                final_scores[pid] = final_scores.get(pid, 0) + (norm_score * weights[0])

        # 处理共被引 (CC)
        if cc_list:
            max_cc = cc_list[0][1]
            for pid, score in cc_list:
                norm_score = score / max_cc # 归一化到 0-1
                final_scores[pid] = final_scores.get(pid, 0) + (norm_score * weights[1])

        # 3. 排序并输出
        # 将分数还原成 0-100 的整数，方便前端展示
        sorted_res = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(pid, int(score * 100)) for pid, score in sorted_res]