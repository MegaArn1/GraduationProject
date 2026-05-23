import os
# 设置 HuggingFace 镜像地址 (必须在导入 sentence_transformers 前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import time
from tqdm import tqdm

# --- 配置部分 ---
# 输入文件 (上一元生成的包含 Reclassified_Tag 的文件)
INPUT_FILE = 'bilibili_qa_dataset_3.xlsx' 
OUTPUT_FILE = 'bilibili_qa_dataset_micro_clustered.xlsx'

# API 配置 (SiliconFlow)
client = OpenAI(
    api_key="sk-spsvvcfyvesstgkretahwejjztpwwccauweyrsktbjhizpdy",
    base_url="https://api.siliconflow.cn/v1/chat/completions"
)
LLM_MODEL = "Qwen/Qwen3-14B"

# 聚类参数
# distance_threshold: 距离阈值。
# 使用余弦距离时，distance = 1 - similarity。
# 如果希望相似度 > 0.6 的归为一类，则 distance < 0.4。
DISTANCE_THRESHOLD = 0.4 
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' 
# 指定服务器本地模型路径
# EMBEDDING_MODEL_NAME = '/public/home/kyy_yxu/zzx_models/modelscope/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def call_llm_api(prompt, retries=3):
    """
    调用 LLM API 并尝试解析 JSON 响应
    """
    messages = [
        {"role": "system", "content": "你是一个专业的医学数据分析师。请只输出合法的 JSON 格式。"},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"} # 强制 JSON 输出
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            if attempt == retries - 1:
                print(f"LLM API Error after {retries} attempts: {e}")
                return {
                    "standard_question": "生成失败",
                    "key_answer_points": "生成失败"
                }
            time.sleep(2)

def generate_cluster_summary(intents_list, answers_list):
    """
    生成聚类总结Prompt并调用API
    """
    # 选取最多 20 个意图作为代表，防止 Token 溢出
    intents_text = "\n".join([f"- {str(i)}" for i in intents_list[:20]])
    
    # 选取对应的回答进行总结，同样防止溢出，并处理可能的空值
    answers_clean = [str(a) for a in answers_list[:20] if pd.notna(a) and str(a).strip() != ""]
    if not answers_clean:
        answers_text = "(该组数据暂无有效回答内容)"
    else:
        answers_text = "\n".join([f"- {a}" for a in answers_clean])
    
    prompt = f"""
    你是一个专业的医学数据分析师。以下是一组用户关于“青少年生殖健康”的提问意图以及模型针对这些问题生成的详细回答，它们被算法判定为语义相似。
    
    【用户提问意图】：
    {intents_text}
    
    【模型生成的现有回答】：
    {answers_text}
    
    请完成以下任务：
    1. **归纳核心问题 (Standard Question)**：用一句通顺、标准的医学问句总结这些用户的共同疑惑。
    2. **提炼回答逻辑 (Key Answer Points)**：仔细阅读上述【模型生成的现有回答】，总结这些回答中包含的共同关键点。
       * 警告：请只总结上述提供的回答内容，**不要**根据你自己的知识添加额外信息。你需要忠实还原数据集中现有的回答逻辑。
    
    请务必返回如下格式的 JSON：
    {{
        "standard_question": "这里填核心问题",
        "key_answer_points": "1. 要点一; 2. 要点二; 3. 要点三"
    }}
    """
    return call_llm_api(prompt)

def main():
    # 1. 数据准备
    print(f"正在读取 {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}。请确保上一处理步骤已完成或修改文件名。")
        return

    df = pd.read_excel(INPUT_FILE)
    
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
        # 兼容 boolean True 和字符串 "TRUE"/"True"
        df = df[df['is_valid_question'].astype(str).str.upper() == 'TRUE']
        print(f"根据 is_valid_question 筛选有效问题: {original_count} -> {len(df)}")
    else:
        print("警告：未找到 'is_valid_question' 列，跳过有效性筛选。")
    
    print(f"正在加载 Embedding 模型 ({EMBEDDING_MODEL_NAME})... (首次运行可能需要下载)")
    try:
        # 显式指定设备，防止多 GPU 环境下的索引错误
        if torch.cuda.is_available():
            # 强制使用第一个可见的 GPU
            device = "cuda:0"
            gpu_count = torch.cuda.device_count()
            print(f"检测到 CUDA 可用，可见 GPU 数量: {gpu_count}。将使用设备: {device}")
        else:
            device = "cpu"
            print("未检测到 CUDA，将使用 CPU。")

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print(f"模型加载成功，运行设备: {embedder.device}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用更基础的模型 'all-MiniLM-L6-v2'...")
        # 同样显式指定设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"基础模型加载成功，运行设备: {embedder.device}")

    results = []
    
    # 获取所有的宏观分类
    unique_categories = df['Reclassified_Tag'].unique()
    print(f"检测到 {len(unique_categories)} 个宏观分类，准备开始微观聚类...")

    # 3. 聚类主循环
    # 为了适应 Slurm/集群环境日志查看，移除了 tqdm 进度条，改用 flush=True 的显式打印
    for idx, category in enumerate(unique_categories):
        print(f"\n[{idx+1}/{len(unique_categories)}] 正在处理分类: {category}", flush=True)
        
        # 获取当前类别的子集
        subset = df[df['Reclassified_Tag'] == category].copy()
        print(f"  - 数据量: {len(subset)}", flush=True)
        
        # 如果数据量太少，直接归为一个簇
        if len(subset) < 2:
            cluster_labels = [0] * len(subset)
            num_clusters = 1
            print("  - 数据量过少，直接归为 1 个簇。", flush=True)
        else:
            # 向量化
            # print("  - 正在进行向量化...")
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
        
        # 将聚类标签添加回子集
        subset['cluster_id'] = cluster_labels
        
        # 4. LLM 总结主循环 (针对当前分类下的每个簇)
        # 移除 tqdm，改为周期性打印日志，确保在 Slurm 输出文件中可见
        print(f"  - 开始生成 LLM 总结 (共 {num_clusters} 个簇)...", flush=True)
        
        for c_id in range(num_clusters):
            # 每处理 5 个簇打印一次进度，反馈更及时
            if (c_id + 1) % 5 == 0 or (c_id + 1) == num_clusters:
                print(f"    > 正在处理簇 {c_id + 1}/{num_clusters}...", flush=True)

            cluster_data = subset[subset['cluster_id'] == c_id]
            intents_list = cluster_data['intent_summary'].tolist()
            answers_list = cluster_data['ans_detail'].tolist()
            
            # 调用 LLM 生成总结
            llm_result = generate_cluster_summary(intents_list, answers_list)
            
            # 选取样本意图 (用于人工核对)
            sample_intents = "; ".join([str(i) for i in intents_list[:3]])
            
            results.append({
                "category": category,
                "cluster_internal_id": c_id,
                "standard_question": llm_result.get("standard_question", ""),
                "key_answer_points": llm_result.get("key_answer_points", ""),
                "sample_intents": sample_intents,
                "total_count": len(cluster_data),
                "original_ids": ",".join(map(str, cluster_data['id'].tolist())) # 如果需要将来回溯
            })

    # 5. 输出
    print(f"正在保存聚类结果到 {OUTPUT_FILE}...")
    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_FILE, index=False)
    print("完成！")
    print(f"共生成 {len(result_df)} 个微观意图簇。")

if __name__ == "__main__":
    main()
