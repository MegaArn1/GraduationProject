import os
# 设置 HuggingFace 镜像地址 (必须在导入 sentence_transformers 前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import time

# --- 配置部分 ---
INPUT_FILE = 'bilibili_qa_dataset_3.xlsx' 
# 这一步输出的中间文件，包含 cluster_id
INTERMEDIATE_FILE = 'bilibili_qa_dataset_clustered_raw.xlsx'

# 聚类参数
DISTANCE_THRESHOLD = 0.4 
# 指定服务器本地模型路径，如果本地没有则会自动尝试下载（请根据实际环境修改路径）
EMBEDDING_MODEL_NAME = '/public/home/kyy_yxu/zzx_models/modelscope/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def main():
    # 1. 数据准备
    print(f"正在读取 {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}。")
        return

    df = pd.read_excel(INPUT_FILE)
    
    # 仅从原始数据中读取关键列，保留 id 以便后续追踪
    if 'id' not in df.columns:
        # 如果没有 id 列，创建一个临时 id
        df['id'] = range(len(df))

    required_cols = ['Reclassified_Tag', 'intent_summary', 'standardized_question', 'ans_detail']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：输入文件缺少必要列 '{col}'")
            return

    # 过滤掉 intent_summary 为空的数据
    df = df.dropna(subset=['intent_summary']).copy()
    
    # 过滤 is_valid_question
    if 'is_valid_question' in df.columns:
        original_count = len(df)
        df = df[df['is_valid_question'].astype(str).str.upper() == 'TRUE']
        print(f"根据 is_valid_question 筛选有效问题: {original_count} -> {len(df)}")
    else:
        print("警告：未找到 'is_valid_question' 列，跳过有效性筛选。")
    
    # 初始化 cluster_id 列
    df['cluster_id'] = -1

    # 2. 加载模型
    print(f"正在加载 Embedding 模型 ({EMBEDDING_MODEL_NAME})...")
    try:
        if torch.cuda.is_available():
            device = "cuda:0"
            gpu_count = torch.cuda.device_count()
            print(f"检测到 CUDA 可用，可见 GPU 数量: {gpu_count}。将使用设备: {device}")
        else:
            device = "cpu"
            print("未检测到 CUDA，将使用 CPU。")

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print(f"模型加载成功，运行设备: {embedder.device}")
        
    except Exception as e:
        print(f"加载指定的本地模型失败: {e}")
        print("尝试使用更基础的模型 'all-MiniLM-L6-v2' (如果能联网会尝试下载)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"基础模型加载成功，运行设备: {embedder.device}")

    # 获取所有的宏观分类
    unique_categories = df['Reclassified_Tag'].unique()
    print(f"检测到 {len(unique_categories)} 个宏观分类，准备开始微观聚类...", flush=True)

    # 3. 聚类主循环
    for idx, category in enumerate(unique_categories):
        print(f"\n[{idx+1}/{len(unique_categories)}] 正在处理分类: {category}", flush=True)
        
        # 获取当前类别的子集掩码
        mask = df['Reclassified_Tag'] == category
        subset = df[mask].copy()
        
        print(f"  - 数据量: {len(subset)}", flush=True)
        
        if len(subset) < 2:
            cluster_labels = [0] * len(subset)
            num_clusters = 1
            print("  - 数据量过少，直接归为 1 个簇。", flush=True)
        else:
            # 向量化
            intents = subset['intent_summary'].tolist()
            embeddings = embedder.encode(intents)
            
            # 归一化向量
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='euclidean', 
                linkage='average', 
                distance_threshold=DISTANCE_THRESHOLD
            )
            cluster_labels = clustering.fit_predict(normalized_embeddings)
            num_clusters = clustering.n_clusters_
            print(f"  - 聚类完成，生成 {num_clusters} 个微观簇。", flush=True)
        
        # 将生成的 cluster_id 回填到原始 DataFrame 中
        # 注意：这里的 cluster_id 是每个分类内部的相对 ID (0, 1, 2...)
        # 为了全局区分，我们通常不需要全局唯一的 cluster_id，只需 (category, cluster_id) 组合唯一即可
        df.loc[mask, 'cluster_id'] = cluster_labels
    
    # 4. 保存中间结果
    print(f"\n正在保存聚类结果到 {INTERMEDIATE_FILE}...", flush=True)
    df.to_excel(INTERMEDIATE_FILE, index=False)
    print("Step 1 完成！请运行 Step 2 脚本进行 LLM 总结。", flush=True)

if __name__ == "__main__":
    main()
