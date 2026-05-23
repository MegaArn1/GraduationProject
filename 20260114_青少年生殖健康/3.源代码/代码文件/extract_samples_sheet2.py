import pandas as pd
import os

# 配置
INPUT_FILE = 'bilibili_qa_dataset_3.xlsx'  # 原数据集
OUTPUT_FILE = 'bilibili_qa_dataset_micro_clustered.xlsx' # 目标文件
SHEET_NAME = 'Sheet2'
SAMPLES_PER_CATEGORY = 10
RANDOM_SEED = 42

def main():
    print(f"正在读取 {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}。")
        return

    df = pd.read_excel(INPUT_FILE)
    
    # 检查列
    # 如果没有id列，使用index作为id
    if 'id' not in df.columns:
         df['id'] = df.index
         
    needed_cols = ['id', 'Reclassified_Tag', 'standardized_question', 'ans_detail', 'is_valid_question']
    for col in needed_cols:
        if col not in df.columns:
            print(f"错误：缺少列 {col}")
            print(f"当前列: {df.columns.tolist()}")
            return

    # 1. 筛选 is_valid_question 为 TRUE
    # 兼容不区分大小写和布尔值
    valid_mask = df['is_valid_question'].astype(str).str.upper() == 'TRUE'
    valid_df = df[valid_mask].copy()
    
    print(f"有效问题筛选结果: {len(df)} -> {len(valid_df)}")
    
    # 2. 按 Reclassified_Tag 分组随机抽取
    sampled_frames = []
    # 确保 Reclassified_Tag 没有空值
    valid_df = valid_df.dropna(subset=['Reclassified_Tag'])
    
    groups = valid_df.groupby('Reclassified_Tag')
    
    print("\n抽取详情:")
    for name, group in groups:
        # 如果不够 10 条，就取全部；否则随机取 10 条
        n = min(len(group), SAMPLES_PER_CATEGORY)
        sampled = group.sample(n=n, random_state=RANDOM_SEED)
        sampled_frames.append(sampled)
        print(f"  - 分类 '{name}': 抽取 {n} 条 (总数 {len(group)})")
        
    if not sampled_frames:
        print("没有抽取到任何数据。")
        return
        
    result_df = pd.concat(sampled_frames)
    
    # 3. 只保留特定列
    final_cols = ['id', 'Reclassified_Tag', 'standardized_question', 'ans_detail']
    result_df = result_df[final_cols]
    
    print(f"\n总计抽取 {len(result_df)} 条数据。")

    # 4. 写入 Excel 的 Sheet2
    print(f"正在写入 {OUTPUT_FILE} 的 {SHEET_NAME}...")
    
    if os.path.exists(OUTPUT_FILE):
        try:
            # 尝试以追加模式打开
            with pd.ExcelWriter(OUTPUT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                result_df.to_excel(writer, sheet_name=SHEET_NAME, index=False)
            print("写入成功！")
        except Exception as e:
            print(f"追加写入失败: {e}")
            print("尝试读取这整个文件，然后在内存中添加Sheet再保存...")
            try:
                # 读取现有所有 sheets
                sheets_dict = pd.read_excel(OUTPUT_FILE, sheet_name=None)
                sheets_dict[SHEET_NAME] = result_df
                
                # 重新写入所有 sheets
                with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
                    for s_name, s_df in sheets_dict.items():
                        s_df.to_excel(writer, sheet_name=s_name, index=False)
                print("覆盖重写成功！")
            except Exception as e2:
                print(f"重写也失败了: {e2}")
    else:
        # 文件不存在，直接创建
        result_df.to_excel(OUTPUT_FILE, sheet_name=SHEET_NAME, index=False)
        print("新文件创建成功！")

if __name__ == "__main__":
    main()
