import argparse
import pandas as pd
import requests
import json
import time
from tqdm import tqdm

# ================= 配置区域 =================
# API 配置
API_KEY = "sk-spsvvcfyvesstgkretahwejjztpwwccauweyrsktbjhizpdy" # 你提供的密钥
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1-Terminus" 
# 注意：SiliconFlow 目前主要支持 "deepseek-ai/DeepSeek-V3" 或 "deepseek-ai/DeepSeek-R1"
# 如果 "DeepSeek-V3.1-Terminus" 报错，请尝试 "deepseek-ai/DeepSeek-V3"

# 输入输出文件路径
# INPUT_FILE = "ex_ernie-speed-video_crb_filtered_classify_1_to_5113_v1.3.xlsx"  # 假设你的数据保存为Excel
INPUT_FILE = "bilibili_question_regulated_test-1.xlsx"
OUTPUT_FILE = "bilibili_question_regulated_4.xlsx"

# ================= 提示词设计 (Prompt Design) =================
def construct_prompt(video_title, comment_content):
    """
    构建发送给大模型的提示词。
    依据开题报告：将口语化、非正式的问题进行规范化重写，使其表达更为正式、规范和科学。
    """
    system_prompt = """
你是一名专业的青少年生殖健康教育专家和医学编辑。
你的任务是处理来自社交媒体（Bilibili）的用户评论，将其中包含的非正式、口语化的健康疑问，转化为“科学、规范、标准”的医学问答题目。

请遵循以下原则：
1. **识别问题**：判断评论是否包含实质性的生殖健康问题。如果是纯粹的情绪发泄、无关玩笑或无意义内容，标记为非问题。
2. **术语规范化**：将网络用语转化为医学术语（例如：“套套”->“避孕套/安全套”，“大姨妈”->“月经”，“有了”->“怀孕”）。
3. **去语境化与去噪**：去除表情符号、语气词（如“咋办”、“救命”）、具体的个人琐碎细节（除非对病情判断必要），使问题具有普适性和教育价值。
4. **科学重写**：将问题改写为通顺、客观的疑问句。

请以 JSON 格式返回结果，不要包含Markdown格式标记：
{
    "is_question": true/false,  // 是否包含健康问题
    "original_intent": "简要概括用户意图",
    "rewritten_question": "规范化后的标准问题" // 如果不是问题，留空
}
"""

    user_prompt = f"""
【上下文信息】
视频标题：{video_title}
用户评论原始内容：{comment_content}

请根据上述内容生成JSON响应。
"""
    return system_prompt, user_prompt

# ================= API 调用函数 =================
def call_deepseek_api(video_title, comment_content):
    system_prompt, user_prompt = construct_prompt(video_title, comment_content)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.3, # 低温度以保证输出的严谨性和确定性
        "response_format": {"type": "json_object"} # 强制返回JSON
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 解析 content
        content_str = result['choices'][0]['message']['content']
        return json.loads(content_str)
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return {
            "is_question": False,
            "original_intent": "Error",
            "rewritten_question": ""
        }

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="规范化评论问题，可通过 -l start end 只处理特定行 (1-based, inclusive)"
    )
    parser.add_argument(
        "-l",
        "--line-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="指定要处理的起止行号，示例：-l 100 200 表示处理第100到200行 (含)"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="仅重试 intent_summary 为 Error 的行"
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=f"输入文件路径，默认使用配置中的 {INPUT_FILE}"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出文件路径，默认：重试模式写回输入文件，其它模式写到配置 OUTPUT_FILE"
    )
    args = parser.parse_args()
    if args.line_range:
        start, end = args.line_range
        if start < 1 or end < start:
            parser.error("行号必须为正整数且 START <= END")
    return args


def resolve_target_indices(df, line_range):
    if not line_range:
        return df.index

    start, end = line_range
    max_row = len(df)
    if start > max_row:
        return df.iloc[0:0].index  # empty range

    start_idx = start - 1
    end_idx = min(end, max_row)
    return df.iloc[start_idx:end_idx].index


def filter_error_indices(df, line_range):
    if 'intent_summary' not in df.columns:
        return df.iloc[0:0].index

    mask = df['intent_summary'].astype(str).str.lower() == 'error'
    error_indices = df.index[mask]

    if line_range:
        allowed_indices = resolve_target_indices(df, line_range)
        return error_indices.intersection(allowed_indices)

    return error_indices


def ensure_result_columns(df):
    if 'is_valid_question' not in df.columns:
        df['is_valid_question'] = False
    if 'standardized_question' not in df.columns:
        df['standardized_question'] = ""
    if 'intent_summary' not in df.columns:
        df['intent_summary'] = ""


# ================= 主处理流程 =================
def main(line_range=None, retry_errors=False, input_file=None, output_file=None):
    # 1. 读取数据
    # 这里为了演示，我手动创建一个DataFrame，实际使用时请用 pd.read_excel(INPUT_FILE)
    print("正在读取数据...")

    
    # 如果你有真实文件，取消下面注释：
    input_path = input_file or INPUT_FILE
    df = pd.read_excel(input_path)

    # 2. 准备新列
    ensure_result_columns(df)

    target_indices = filter_error_indices(df, line_range) if retry_errors else resolve_target_indices(df, line_range)
    if retry_errors:
        print(f"重试模式：检测到 {len(target_indices)} 行 intent_summary=Error，将重新请求。")
    elif line_range:
        start, end = line_range
        if len(target_indices) == 0:
            print(f"警告：指定的行范围 {start}-{end} 超出数据总行数 {len(df)}，不会处理任何记录。")
        else:
            print(f"仅处理第 {start}-{end} 行，共 {len(target_indices)} 行数据。")

    # 3. 遍历处理
    print("开始调用 DeepSeek 进行问题规范化...")
    iterable_indices = target_indices if (line_range or retry_errors) else df.index
    for index in tqdm(iterable_indices, total=len(iterable_indices)):
        row = df.loc[index]
        title = row['Title']
        content = row['Content']
        
        # 简单的预清洗：如果内容太短或为空，跳过
        if pd.isna(content) or len(str(content)) < 2:
            continue
            
        # 调用 API
        result = call_deepseek_api(title, content)
        
        # 保存结果
        df.at[index, 'is_valid_question'] = result.get('is_valid_question', result.get('is_question', False))
        df.at[index, 'standardized_question'] = result.get('rewritten_question', "")
        df.at[index, 'intent_summary'] = result.get('original_intent', "")
        
        # 避免触发 API 速率限制 (Rate Limit)，适当休眠
        time.sleep(0.5)

    # 4. 导出结果
    print("处理完成，正在保存...")
    # 保存所有结果
    output_path = output_file
    if not output_path:
        output_path = input_path if retry_errors else OUTPUT_FILE
    df.to_excel(output_path, index=False)
    
    # 打印预览
    print("\n=== 处理结果预览 ===")
    print(df[['Content', 'standardized_question']].head().to_markdown(index=False))

if __name__ == "__main__":
    # 运行示例：python question_regulation.py -l 100 200 只处理第100-200行
    # 重试示例：python question_regulation.py --retry-errors --input-file bilibili_question_regulated_4.xlsx
    args = parse_cli_args()
    main(
        line_range=args.line_range,
        retry_errors=args.retry_errors,
        input_file=args.input_file,
        output_file=args.output_file
    )