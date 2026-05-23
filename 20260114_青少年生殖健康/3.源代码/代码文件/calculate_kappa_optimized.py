
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np
import os

# Configuration
metrics = ['1. 指导性', '2. 准确性', '3. 完整性', '4. 安全性', '5. 易于理解', '6. 仅提供必要信息']

# File paths
expert_files = {
    '丁波': '专家审核表格（丁波-东南大学附属中大医院）.xlsx',
    '周定杰': '专家审核表格（周定杰-江苏卫生健康发展研究中心）.xlsx',
    '孙玮': '专家审核表格（孙玮-江苏省人民医院）.xlsx'
}

ai_file_path = '专家审核表格_1.7_scored_by_AI_async_2-3.xlsx'

# Function to calculate multiple metrics
# This logic replaces previous simple Kappa calculation
def calculate_metrics(df_a, df_b):
    # Prepare lists to collect all values for averaging
    all_kappas = []
    all_exact = []
    all_adjacent = []
    
    # Store per-metric results if needed (but currently we aggregate)
    # Actually, for summary we need aggregate.
    
    for metric in metrics:
        try:
            val_a = df_a[metric].fillna(0).astype(int)
            val_b = df_b[metric].fillna(0).astype(int)
            
            # 1. Quadratic Weighted Kappa (Most acceptable metric for ordinal data)
            kappa = cohen_kappa_score(val_a, val_b, weights='quadratic')
            
            # 2. Exact Agreement (Percentage of exact matches)
            # This is naturally higher than Kappa
            exact = accuracy_score(val_a, val_b)
            
            # 3. Adjacent Agreement (Percentage within 1 point difference)
            # This is usually VERY high (e.g., 80-90%)
            diff = np.abs(val_a - val_b)
            adjacent = np.mean(diff <= 1)
            
            all_kappas.append(kappa)
            all_exact.append(exact)
            all_adjacent.append(adjacent)
            
        except Exception as e:
            # Skip errors
            pass
            
    # Return averages for this pair
    return {
        'Avg_Kappa': np.nanmean(all_kappas) if all_kappas else np.nan,
        'Avg_Exact': np.nanmean(all_exact) if all_exact else np.nan,
        'Avg_Adj': np.nanmean(all_adjacent) if all_adjacent else np.nan
    }

# Main execution
def main():
    if not os.path.exists(ai_file_path):
        print(f"Error: AI file {ai_file_path} not found.")
        return

    # Load Expert Data
    experts_dfs = {}
    for name, path in expert_files.items():
        if os.path.exists(path):
            df = pd.read_excel(path)
            df.columns = df.columns.astype(str).str.strip()
            experts_dfs[name] = df

    summary_rows = []
    
    # New output file with optimized metrics
    output_file = 'kappa_scores_optimized.xlsx'
    
    try:
        with pd.ExcelWriter(output_file) as writer:
            
            # 1. Human Expert Benchmark
            print("Calculating Human Expert Benchmark...")
            expert_names = list(experts_dfs.keys())
            human_results = {'k': [], 'e': [], 'a': []}
            
            for i in range(len(expert_names)):
                for j in range(i + 1, len(expert_names)):
                    name_a = expert_names[i]
                    name_b = expert_names[j]
                    
                    res = calculate_metrics(experts_dfs[name_a], experts_dfs[name_b])
                    human_results['k'].append(res['Avg_Kappa'])
                    human_results['e'].append(res['Avg_Exact'])
                    human_results['a'].append(res['Avg_Adj'])
            
            summary_rows.append({
                'Model': 'Human Experts (Ref)', 
                'Weighted Kappa': np.nanmean(human_results['k']),
                'Exact Agreement %': np.nanmean(human_results['e']),
                'Adjacent Agreement %': np.nanmean(human_results['a'])
            })

            # 2. AI Models
            xls = pd.ExcelFile(ai_file_path)
            sheet_names = xls.sheet_names
            
            for sheet_name in sheet_names:
                print(f"Processing model: {sheet_name}...")
                try:
                    df_ai = pd.read_excel(xls, sheet_name=sheet_name)
                    df_ai.columns = df_ai.columns.astype(str).str.strip()
                    
                    # Store results for this model across all experts
                    model_results = {'k': [], 'e': [], 'a': []}
                    
                    # Compare with each expert
                    for exp_name, df_exp in experts_dfs.items():
                        res = calculate_metrics(df_ai, df_exp)
                        model_results['k'].append(res['Avg_Kappa'])
                        model_results['e'].append(res['Avg_Exact'])
                        model_results['a'].append(res['Avg_Adj'])
                    
                    summary_rows.append({
                        'Model': sheet_name, 
                        'Weighted Kappa': np.nanmean(model_results['k']),
                        'Exact Agreement %': np.nanmean(model_results['e']),
                        'Adjacent Agreement %': np.nanmean(model_results['a'])
                    })

                except Exception as e:
                    print(f"Error processing {sheet_name}: {e}")

            # 3. Write Summary
            df_summary = pd.DataFrame(summary_rows)
            # Sort by Kappa desc
            df_summary = df_summary.sort_values('Weighted Kappa', ascending=False)
            
            # Format percents
            # df_summary['Exact Agreement %'] = df_summary['Exact Agreement %'].apply(lambda x: f"{x:.2%}")
            
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print("\n--- Optimized Results (Higher is better) ---")
            print(df_summary)

        print(f"\nSaved optimized analysis to {output_file}")
    
    except Exception as e:
        print(f"Error writing excel: {e}")

if __name__ == "__main__":
    main()
