import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm

# 配置API
client = OpenAI(
    api_key="sk-spsvvcfyvesstgkretahwejjztpwwccauweyrsktbjhizpdy",
    base_url="https://api.siliconflow.cn/v1"
)

def classify_question(question, categories, fallback_tag="其他内容"):
    categories_str = ", ".join(categories)
    prompt = f"""
    你是一个仔细的数据标注员。请根据以下问题内容，判断其是否属于给定的类别之一。
    
    问题："{question}"
    
    可选类别：[{categories_str}]
    
    如果不属于上述任何类别，请返回“{fallback_tag}”。
    请直接返回类别名称，不要包含任何其他文字。
    """
    
    for retries in range(3):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-14B",
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。请只输出类别名称。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # 低温度以增加确定性
                max_tokens=50,
                timeout=30 # 增加超时设置
            )
            result = response.choices[0].message.content.strip()
            
            # 简单清洗结果，防止模型输出额外标点
            for cat in categories:
                if cat in result:
                    return cat
            if fallback_tag in result or "其他" in result or "无关" in result:
                return fallback_tag
                
            return result 
        except Exception as e:
            if retries == 2:
                print(f"Error processing question: {question[:20]}... Error: {e}")
                return "API_ERROR"
            time.sleep(2) # 等待后重试
    return "API_ERROR"

def main():
    # 读取Excel文件
    # 假设文件名为 bilibili_question_regulated-full.xlsx，如果不是请修改此处
    input_file = 'bilibili_qa_dataset_2.xlsx'
    output_file = 'bilibili_qa_dataset_3.xlsx'
    
    print(f"正在读取 {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"找不到文件 {input_file}。尝试读取 workspace 中的其他 xlsx 文件...")
        import glob
        files = glob.glob("*.xlsx")
        if not files:
            print("没有找到Excel文件，请确认文件名。")
            return
        input_file = files[0] # 取第一个
        print(f"读取 {input_file}...")
        df = pd.read_excel(input_file)

    if 'Tag' not in df.columns or 'standardized_question' not in df.columns:
        print("错误：文件中缺少 'Tag' 或 'standardized_question' 列。")
        print("当前列：", df.columns)
        return

    # 获取所有非“其他内容”或“无关内容”的类别
    all_tags = df['Tag'].dropna().unique().tolist()
    
    # 定义我们要重新分类的目标标签
    target_tag = "其他内容"
    if "其他内容" not in all_tags and "无关内容" in all_tags:
        print("提示：未找到 '其他内容' 标签，但在数据中发现了 '无关内容'。将针对 '无关内容' 进行处理。")
        target_tag = "无关内容"
    
    valid_categories = [tag for tag in all_tags if tag != target_tag and tag != "其他内容" and tag != "无关内容"]
    
    print(f"检测到的有效类别 ({len(valid_categories)}个): {valid_categories}")
    
    # 筛选出 Tag 为目标标签的行
    mask = df['Tag'] == target_tag
    rows_to_process = df[mask]
    
    print(f"共有 {len(rows_to_process)} 行数据标记为 '{target_tag}'，准备重新分类...")
    
    # 创建新列存储重新分类的结果，初始化为原Tag
    if 'Reclassified_Tag' not in df.columns:
        df['Reclassified_Tag'] = df['Tag']
    
    # 对需要处理的行进行迭代
    # 使用 tqdm 显示进度条
    for index, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process)):
        question = row['standardized_question']
        
        # 如果问题为空，跳过
        if pd.isna(question):
            continue
            
        new_tag = classify_question(question, valid_categories, fallback_tag=target_tag)
        
        # 更新DataFrame
        df.at[index, 'Reclassified_Tag'] = new_tag
        
        # 稍微暂停避免速率限制（根据API限制调整）
        # time.sleep(0.1) 

    # 保存结果
    print(f"正在保存结果到 {output_file}...")
    df.to_excel(output_file, index=False)
    print("完成！")

if __name__ == "__main__":
    main()
