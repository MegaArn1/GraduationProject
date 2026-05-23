'''
03/05
提取的数据，由于一条视频评论对应多个Sub_C_Reply_Content的原因，使得在id不同的情况下评论相关的字段完全相同。
本程序就是为去除重复数据、仅保留id最前的数据而设计。

'''

import pandas as pd

# **加载 Excel 文件**
file_path = "comment_bullet_keywords_extraction.xlsx"

# 读取 "comment_all" 和 "comment_appended" 两个 sheet
df_comment_all = pd.read_excel(file_path, sheet_name="comment_all")
df_comment_appended = pd.read_excel(file_path, sheet_name="comment_appended")

# **去重逻辑**
def remove_duplicates(df):
    """删除除 `id` 外其他字段完全重复的行，仅保留 `id` 最小的那条"""
    df_sorted = df.sort_values(by="id")  # 先按 id 升序排序
    df_unique = df_sorted.drop_duplicates(subset=df.columns.difference(["id"]), keep="first")  # 去重
    return df_unique

# 应用去重函数
df_comment_all_cleaned = remove_duplicates(df_comment_all)
df_comment_appended_cleaned = remove_duplicates(df_comment_appended)

# **保存去重后的数据**
output_file = "comment_bullet_keywords_extraction-rm_duplicated.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df_comment_all_cleaned.to_excel(writer, sheet_name="comment_all", index=False)
    df_comment_appended_cleaned.to_excel(writer, sheet_name="comment_appended", index=False)

print(f"✅ 处理完成，去重后的数据已保存至 `{output_file}` 🎉")