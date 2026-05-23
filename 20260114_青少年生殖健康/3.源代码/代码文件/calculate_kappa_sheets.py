import pandas as pd
from sklearn.metrics import cohen_kappa_score
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

# Function to calculate Kappa column
def get_kappa_column(df_a, df_b):
    results = {}
    for metric in metrics:
        try:
            # Check availability
            if metric not in df_a.columns or metric not in df_b.columns:
                 results[metric] = np.nan
                 continue

            val_a = df_a[metric].fillna(0).astype(int)
            val_b = df_b[metric].fillna(0).astype(int)
            
            # Weighted Kappa (Quadratic)
            # kappa_quad = cohen_kappa_score(val_a, val_b, weights='quadratic')
            kappa_quad = cohen_kappa_score(val_a, val_b, weights='linear')  # Linear weights treat differences linearly, Quadratic penalizes larger differences more

            results[metric] = kappa_quad
        except Exception as e:
            results[metric] = np.nan
    return results

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
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            experts_dfs[name] = df
        else:
            print(f"Warning: Expert file {path} not found.")

    summary_data = []
    
    # Create Excel Writer
    output_file = 'kappa_scores_linear.xlsx'
    
    try:
        with pd.ExcelWriter(output_file) as writer:
            
            # 1. Inter-Expert Reliability (Comparison)
            inter_expert_results = {}
            expert_names = list(experts_dfs.keys())
            
            for i in range(len(expert_names)):
                for j in range(i + 1, len(expert_names)):
                    name_a = expert_names[i]
                    name_b = expert_names[j]
                    col_name = f"{name_a} vs {name_b}"
                    
                    # Calculate Kappa for each metric between experts
                    kappa_metrics = {}
                    for m in metrics:
                         try:
                             val_a = experts_dfs[name_a][m].fillna(0).astype(int)
                             val_b = experts_dfs[name_b][m].fillna(0).astype(int)
                             # Linear weights
                             kappa_metrics[m] = cohen_kappa_score(val_a, val_b, weights='linear')
                         except:
                             kappa_metrics[m] = np.nan
                    inter_expert_results[col_name] = kappa_metrics
            
            df_inter = pd.DataFrame(inter_expert_results)
            df_inter.loc['Average'] = df_inter.mean()
            df_inter.to_excel(writer, sheet_name='Inter-Expert')
            
            # Add pure expert average to summary
            avg_inter_val = df_inter.loc['Average'].mean()
            expert_summary_row = {'Model': 'Human Experts (Ref)', 'Average Kappa': avg_inter_val}
            for m in metrics:
                if m in df_inter.index:
                    expert_summary_row[m] = df_inter.loc[m].mean()
            summary_data.append(expert_summary_row)


            # 2. Iterate through Sheets in AI file
            xls = pd.ExcelFile(ai_file_path)
            sheet_names = xls.sheet_names
            print(f"Found sheets (Models): {sheet_names}")

            for sheet_name in sheet_names:
                print(f"Processing model: {sheet_name}...")
                try:
                    df_ai = pd.read_excel(xls, sheet_name=sheet_name)
                    df_ai.columns = df_ai.columns.astype(str).str.strip()
                    
                    # Check metrics exist
                    missing = [m for m in metrics if m not in df_ai.columns]
                    if missing:
                        print(f"  Warning: {sheet_name} missing columns: {missing}")
                    
                    # Compare with each expert
                    current_model_results = {}
                    for exp_name, df_exp in experts_dfs.items():
                        col_name = f"vs {exp_name}"
                        
                        # Direct comparison (assuming alignment as checked)
                        kappa_metrics = {}
                        for m in metrics:
                             try:
                                 if m in df_ai.columns and m in df_exp.columns:
                                     val_a = df_ai[m].fillna(0).astype(int)
                                     val_b = df_exp[m].fillna(0).astype(int)
                                     # Linear weights
                                     kappa_metrics[m] = cohen_kappa_score(val_a, val_b, weights='linear')
                                 else:
                                     kappa_metrics[m] = np.nan
                             except:
                                 kappa_metrics[m] = np.nan
                        
                        current_model_results[col_name] = kappa_metrics
                    
                    df_model = pd.DataFrame(current_model_results)
                    
                    # Add Average column for this model (across 3 experts)
                    df_model['Avg (All Experts)'] = df_model.mean(axis=1)
                    
                    # Add Average row (across metrics)
                    overall_avg = df_model['Avg (All Experts)'].mean()
                    
                    # Save to sheet (sanitize sheet name)
                    safe_sheet_name = sheet_name[:30].replace('/', '_').replace('\\', '_')
                    df_model.to_excel(writer, sheet_name=safe_sheet_name)
                    
                    # Parse data for Summary Sheet
                    avg_col = df_model['Avg (All Experts)']
                    
                    row_data = {'Model': sheet_name}
                    row_data['Average Kappa'] = overall_avg
                    
                    for m in metrics:
                        if m in avg_col.index:
                            row_data[m] = avg_col.loc[m]
                    
                    summary_data.append(row_data)

                except Exception as e:
                    print(f"Error processing {sheet_name}: {e}")

            # 3. Write Summary Sheet
            df_summary = pd.DataFrame(summary_data)
            desired_cols = ['Model', 'Average Kappa'] + metrics
            final_cols = [c for c in desired_cols if c in df_summary.columns]
            df_summary = df_summary[final_cols] if not df_summary.empty else df_summary
            
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print("\n--- Summary Table ---")
            print(df_summary)

        print(f"\nSaved multi-model analysis to {output_file}")
    
    except Exception as e:
        print(f"Error writing excel: {e}")

if __name__ == "__main__":
    main()
