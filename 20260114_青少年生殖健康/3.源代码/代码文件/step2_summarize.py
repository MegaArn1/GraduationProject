import pandas as pd
from openai import OpenAI
import json
import time
import os
from tqdm import tqdm

# --- 配置部分 ---
# 输入文件 (Step 1 的输出)
INPUT_FILE = 'bilibili_qa_dataset_clustered_raw_1.xlsx'
OUTPUT_FILE = 'bilibili_qa_dataset_micro_clustered.xlsx'

# API 配置 (SiliconFlow)
client = OpenAI(
    api_key="sk-spsvvcfyvesstgkretahwejjztpwwccauweyrsktbjhizpdy",
    base_url="https://api.siliconflow.cn/v1"
)
LLM_MODEL = "Qwen/Qwen3-14B"
# 每个大类抽取的簇数量（设置为 None 则处理所有簇）
CLUSTERS_PER_CATEGORY_LIMIT = 10 

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
                max_tokens=600,
                response_format={"type": "json_object"}
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
    intents_text = "\n".join([f"- {str(i)}" for i in intents_list[:20]])
    
    # 选取对应的回答进行总结，防止溢出，并处理可能的空值
    answers_clean = [str(a) for a in answers_list[:20] if pd.notna(a) and str(a).strip() != ""]
    if not answers_clean:
        answers_text = "(该组数据暂无有效回答内容)"
    else:
        answers_text = "\n".join([f"- {a}" for a in answers_clean])
    
    prompt = f"""
    你是一个专业的医学数据分析师。以下是一组用户关于“青少年生殖健康”的提问意图以及模型针对这些问题生成的详细回答，它们被算法判定为语义相似。
    
    【用户提问意图 (Intent)】：
    {intents_text}
    
    【模型生成的现有回答 (Existing Answer Details)】：
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
    print(f"正在读取 {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}。请先运行 step1_cluster.py。")
        return

    df = pd.read_excel(INPUT_FILE)
    
    if 'cluster_id' not in df.columns:
        print("错误：输入文件中缺少 'cluster_id' 列。")
        return

    results = []
    
    unique_categories = df['Reclassified_Tag'].unique()
    print(f"检测到 {len(unique_categories)} 个宏观分类，准备进行 LLM 总结...")

    # 为了方便统计进度，我们将所有的 (category, cluster_id) 组合先提取出来
    # 生成待处理任务列表
    tasks = []
    
    # 随机种子，确保每次抽取的簇相对固定（可以根据需要设为 None）
    RANDOM_SEED = 42
    
    for category in unique_categories:
        cat_mask = df['Reclassified_Tag'] == category
        subset = df[cat_mask]
        
        # 获取该类别下所有的微观簇 ID
        unique_cluster_ids = pd.Series(subset['cluster_id'].dropna().unique())
        
        print(f"  - 分类 '{category}' 共有 {len(unique_cluster_ids)} 个微观簇")
        
        # 抽样逻辑
        if CLUSTERS_PER_CATEGORY_LIMIT is not None and len(unique_cluster_ids) > CLUSTERS_PER_CATEGORY_LIMIT:
            selected_cluster_ids = unique_cluster_ids.sample(n=CLUSTERS_PER_CATEGORY_LIMIT, random_state=RANDOM_SEED).sort_values()
            print(f"    -> 已随机抽取 {CLUSTERS_PER_CATEGORY_LIMIT} 个进行处理")
        else:
            selected_cluster_ids = unique_cluster_ids.sort_values()
        
        for c_id in selected_cluster_ids:
            tasks.append({
                'category': category,
                'cluster_id': c_id
            })
            
    print(f"共生成 {len(tasks)} 个微观意图簇待总结。")

    # 使用 tqdm 显示总体进度
    for task in tqdm(tasks, desc="Summarizing Clusters"):
        category = task['category']
        c_id = task['cluster_id']
        
        # 筛选对应的数据
        cluster_data = df[(df['Reclassified_Tag'] == category) & (df['cluster_id'] == c_id)]
        
        intents_list = cluster_data['intent_summary'].tolist()
        answers_list = cluster_data['ans_detail'].tolist()
        
        # 调用 LLM 生成总结
        llm_result = generate_cluster_summary(intents_list, answers_list)
        
        # 选取样本意图
        sample_intents = "; ".join([str(i) for i in intents_list[:3]])
        
        results.append({
            "category": category,
            "cluster_internal_id": c_id,
            "standard_question": llm_result.get("standard_question", ""),
            "key_answer_points": llm_result.get("key_answer_points", ""),
            "sample_intents": sample_intents,
            "total_count": len(cluster_data),
            "original_ids": ",".join(map(str, cluster_data['id'].tolist()))
        })

    # 输出最终结果（这里只保存 summary 表，不想包含详细原始数据的可以只存这个）
    print(f"正在保存最终总结结果到 {OUTPUT_FILE}...")
    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_FILE, index=False)
    print("完成！")

if __name__ == "__main__":
    main()
