import argparse
import pandas as pd
import requests
import json
import time
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-spsvvcfyvesstgkretahwejjztpwwccauweyrsktbjhizpdy" # 你的密钥
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1-Terminus" 

# 输入文件是上一步生成的包含规范化问题的文件
INPUT_FILE = "bilibili_question_regulated-full.xlsx"
OUTPUT_FILE = "bilibili_qa_dataset_test.xlsx"

# ================= 提示词构建 =================
def construct_answer_prompt(question):
    system_prompt = """
你是一位拥有丰富临床经验的“青少年生殖健康专家”和“科普教育者”。你的目标是为青少年提供科学、准确、温暖且易于理解的生殖健康知识。

请根据用户提供的“规范化问题”，生成一份高质量的回答。

【回答原则】
1. **科学准确**：基于世界卫生组织（WHO）及中国权威医疗指南的共识。
2. **通俗易懂**：适合15-24岁人群阅读，解释晦涩术语。
3. **客观中立**：态度包容，去污名化，严禁道德审判。
4. **结构清晰**：逻辑分明。

【输出格式】
请严格以 JSON 格式输出，不要包含Markdown标记，包含以下字段：
{
    "summary": "一句话的核心回答（50字以内），直接给出结论。",
    "detail": "详细的科普解释（200-400字），解释原理。",
    "advice": "具体的行动建议（如：正确避孕方法、就医指引）。",
    "warning": "风险提示（如需立即就医则填写，否则为空）。"
}
"""
    user_prompt = f"""
【待回答的规范化问题】
{question}

请基于上述问题，生成JSON格式的专业科普回答。
"""
    return system_prompt, user_prompt

# ================= API 调用 =================
def generate_answer(question):
    system_prompt, user_prompt = construct_answer_prompt(question)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.5, # 稍微提高一点温度，让科普语言更自然流畅，但不过高以保持准确
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        content_str = result['choices'][0]['message']['content']
        return json.loads(content_str)
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "summary": "生成失败",
            "detail": "",
            "advice": "",
            "warning": ""
        }

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="为规范化问题生成答案，可通过 -l start end 只处理特定行 (1-based, inclusive)"
    )
    parser.add_argument(
        "-l",
        "--line-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="指定要处理的起止行号，例如：-l 10 50 仅处理第10到50行 (含)"
    )
    args = parser.parse_args()
    if args.line_range:
        start, end = args.line_range
        if start < 1 or end < start:
            parser.error("行号必须为正整数且 START <= END")
    return args


def resolve_line_range_indices(df, line_range):
    if not line_range:
        return df.index

    start, end = line_range
    max_row = len(df)
    if start > max_row:
        return df.iloc[0:0].index

    start_idx = start - 1
    end_idx = min(end, max_row)
    return df.iloc[start_idx:end_idx].index


def ensure_answer_columns(df):
    for col in ["ans_summary", "ans_detail", "ans_advice", "ans_warning"]:
        if col not in df.columns:
            df[col] = ""


def truthy_series(series):
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


# ================= 主程序 =================
def main(line_range=None):
    # 1. 读取上一步处理好的数据
    print("正在读取数据...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"找不到文件 {INPUT_FILE}，请先运行上一步的代码。")
        return

    # 2. 过滤出有效的问题
    # 只有被标记为是问题，且有规范化文本的行才需要生成答案
    # 注意：根据上一步代码，布尔值可能在Excel里变成了True/False字符串或0/1，这里做个兼容处理
    ensure_answer_columns(df)

    valid_question_flags = truthy_series(df['is_valid_question'])
    mask = valid_question_flags & (df['standardized_question'].notna()) & (df['standardized_question'] != "")
    if line_range:
        allowed_idx = resolve_line_range_indices(df, line_range)
        mask &= df.index.isin(allowed_idx)

    target_rows = df[mask]
    
    print(f"共有 {len(df)} 条数据，其中 {len(target_rows)} 条为有效问题，准备生成答案...")
    if line_range:
        start, end = line_range
        print(f"仅处理第 {start}-{end} 行范围内的有效问题。")

    # 4. 遍历生成
    for index, row in tqdm(target_rows.iterrows(), total=len(target_rows)):
        question = row['standardized_question']
        
        # 调用大模型
        answer_data = generate_answer(question)
        
        # 填入数据
        df.at[index, 'ans_summary'] = answer_data.get('summary', "")
        df.at[index, 'ans_detail'] = answer_data.get('detail', "")
        df.at[index, 'ans_advice'] = answer_data.get('advice', "")
        df.at[index, 'ans_warning'] = answer_data.get('warning', "")
        
        # 避免速率限制
        time.sleep(0.5)

    # 5. 保存最终数据集
    print("正在保存最终数据集...")
    df.to_excel(OUTPUT_FILE, index=False)
    
    # 6. 展示示例
    print("\n=== 问答对生成示例 ===")
    sample = df[mask].head(1)
    if not sample.empty:
        print(f"问题: {sample.iloc[0]['standardized_question']}")
        print(f"核心回答: {sample.iloc[0]['ans_summary']}")
        print(f"详细解释: {sample.iloc[0]['ans_detail'][:50]}...")
        print(f"建议: {sample.iloc[0]['ans_advice']}")

if __name__ == "__main__":
    # 运行示例：python answer_generation.py -l 50 100 仅处理第50-100行
    args = parse_cli_args()
    main(line_range=args.line_range)