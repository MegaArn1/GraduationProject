import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Configuration
metrics = ['1. 指导性', '2. 准确性', '3. 完整性', '4. 安全性', '5. 易于理解', '6. 仅提供必要信息']

# Load data - Using filenames as relative paths since script will be run in the folder
ai_file = '专家审核表格_1.7_scored_by_AI_async_2-3.xlsx'
exp1_file = '专家审核表格（丁波-东南大学附属中大医院）.xlsx'
exp2_file = '专家审核表格（周定杰-江苏卫生健康发展研究中心）.xlsx'
exp3_file = '专家审核表格（孙玮-江苏省人民医院）.xlsx'

df_ai = pd.read_excel(ai_file)
df_exp1 = pd.read_excel(exp1_file)
df_exp2 = pd.read_excel(exp2_file)
df_exp3 = pd.read_excel(exp3_file)

# Clean column names
df_ai.columns = df_ai.columns.astype(str).str.strip()
df_exp1.columns = df_exp1.columns.astype(str).str.strip()
df_exp2.columns = df_exp2.columns.astype(str).str.strip()
df_exp3.columns = df_exp3.columns.astype(str).str.strip()

# Configuration
metrics = ['1. 指导性', '2. 准确性', '3. 完整性', '4. 安全性', '5. 易于理解', '6. 仅提供必要信息']

# Function to calculate Kappa
def get_kappa_column(df_a, df_b, name_a, name_b):
    results = {}
    for metric in metrics:
        try:
            # Access col using standard name
            # Handle potential spaces or slightly different names? We verified names match exactly.
            val_a = df_a[metric].fillna(0).astype(int)
            val_b = df_b[metric].fillna(0).astype(int)
            
            # Weighted Kappa (Quadratic) is standard for ordinal ratings (1-5)
            # weights='quadratic' means differences are squared. If ratings are closer, penalty is smaller.
            # This is standard for Likert scales.
            kappa_quad = cohen_kappa_score(val_a, val_b, weights='quadratic')
            results[metric] = kappa_quad
        except Exception as e:
            print(f"Error calculating {metric} for {name_a} vs {name_b}: {e}")
            results[metric] = np.nan
    return results

# Calculate all pairs
data = {}

# AI vs Experts
data['AI vs 丁波'] = get_kappa_column(df_ai, df_exp1, 'AI', '丁波')
data['AI vs 周定杰'] = get_kappa_column(df_ai, df_exp2, 'AI', '周定杰')
data['AI vs 孙玮'] = get_kappa_column(df_ai, df_exp3, 'AI', '孙玮')

# Expert vs Expert (Inter-Expert Reliability)
data['丁波 vs 周定杰'] = get_kappa_column(df_exp1, df_exp2, '丁波', '周定杰')
data['丁波 vs 孙玮'] = get_kappa_column(df_exp1, df_exp3, '丁波', '孙玮')
data['周定杰 vs 孙玮'] = get_kappa_column(df_exp2, df_exp3, '周定杰', '孙玮')

# Create DataFrame
result_df = pd.DataFrame(data)

# Add average row
result_df.loc['Average'] = result_df.mean()

print("\n--- Kappa Score Results (Quadratic Weighted) ---")
print(result_df)

# Save
output_file = 'kappa_scores_result.xlsx'
result_df.to_excel(output_file)
print(f"\nSaved results to {output_file}")
