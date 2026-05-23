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

# AI files mapping
# Keys are model names, values are filenames
ai_files = {
    'AI_Base': '专家审核表格_1.7_scored_by_AI.xlsx',
    'AI_Async_2-2': '专家审核表格_1.7_scored_by_AI_async_2-2.xlsx',
    'AI_Async_2-3-2': '专家审核表格_1.7_scored_by_AI_async_2-3-2.xlsx',
    'AI_Async_2-3': '专家审核表格_1.7_scored_by_AI_async_2-3.xlsx'
}

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
            kappa_quad = cohen_kappa_score(val_a, val_b, weights='quadratic')
            results[metric] = kappa_quad
        except Exception as e:
            results[metric] = np.nan
    return results

# Main execution
def main():
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

    # Create Excel Writer
    # Use default engine (openpyxl usually) or specify openpyxl
    output_file = 'kappa_scores_multi_model.xlsx'
    
    # Store summary data list
    summary_data = []
    
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
                    # Only calculate if both dataframes have the metric
                    kappa_metrics = {}
                    for m in metrics:
                         try:
                             val_a = experts_dfs[name_a][m].fillna(0).astype(int)
                             val_b = experts_dfs[name_b][m].fillna(0).astype(int)
                             kappa_metrics[m] = cohen_kappa_score(val_a, val_b, weights='quadratic')
                         except:
                             kappa_metrics[m] = np.nan
                    inter_expert_results[col_name] = kappa_metrics
            
            df_inter = pd.DataFrame(inter_expert_results)
            df_inter.loc['Average'] = df_inter.mean()
            df_inter.to_excel(writer, sheet_name='Inter-Expert')
            
            # Add pure expert average to summary
            avg_inter_val = df_inter.mean().mean() # Mean of all expert pairs
            # Create a summary row for experts
            expert_summary_row = {'Model': 'Human Experts (Ref)', 'Average Kappa': avg_inter_val}
            
            # Add metric specific averages across all pairs
            metric_avgs = df_inter.mean(axis=1)
            for m in metrics:
                if m in metric_avgs.index:
                    expert_summary_row[m] = metric_avgs.loc[m]
            summary_data.append(expert_summary_row)


            # 2. AI vs Experts
            for ai_name, ai_path in ai_files.items():
                if not os.path.exists(ai_path):
                    print(f"Skipping {ai_name}: File not found.")
                    continue
                
                print(f"Processing {ai_name}...")
                try:
                    df_ai = pd.read_excel(ai_path)
                    df_ai.columns = df_ai.columns.astype(str).str.strip()
                    
                    # Handle multiple runs (e.g. 2-3-2 has 5x rows)
                    # If AI rows > Expert rows (approx, accounting for duplicates), aggregate
                    # Use length of one expert as reference (147)
                    ref_len = len(experts_dfs[list(experts_dfs.keys())[0]])
                    
                    if len(df_ai) > ref_len * 1.5:
                        print(f"  Note: {ai_name} has {len(df_ai)} rows, aggregating by '问题'...")
                        # Group by Question and take mean of metrics
                        # Ensure '问题' column exists
                        if '问题' in df_ai.columns:
                            # Select only numeric metrics for mean
                            numeric_cols = [c for c in metrics if c in df_ai.columns]
                            # Group and mean
                            df_ai_agg = df_ai.groupby('问题')[numeric_cols].mean().round().astype(int).reset_index()
                            
                            # We need to map these back to the order of the Expert file
                            # Use the first expert loop to align
                            # But wait, we iterate experts inside.
                            # So here, we just prepare a mapped version.
                            # Better strategy: Align inside the expert loop.
                            pass
                        else:
                            print(f"  Warning: '问题' column missing for aggregation in {ai_name}")
                            continue

                    # Check metrics exist
                    missing = [m for m in metrics if m not in df_ai.columns]
                    if missing:
                        print(f"  Warning: {ai_name} missing columns: {missing}")
                    
                    # Compare with each expert
                    current_model_results = {}
                    for exp_name, df_exp in experts_dfs.items():
                        col_name = f"vs {exp_name}"
                        
                        # Calculate Kappa for current AI vs Expert
                        kappa_metrics = {}
                        
                        # Prepare aligned data
                        # If AI file was aggregated (or needs to be aligned)
                        # We use expert dataframe as reference for rows
                        df_exp_aligned = df_exp.copy()
                        
                        # Left merge with AI scores on '问题' if AI has different structure
                        if '问题' in df_ai.columns and '问题' in df_exp_aligned.columns:
                            # If AI has many duplicates (2-3-2), group first
                            if len(df_ai) > len(df_exp) * 1.5:
                                numeric_cols = [c for c in metrics if c in df_ai.columns]
                                df_ai_grouped = df_ai.groupby('问题')[numeric_cols].mean().round().reset_index()
                                df_merged = pd.merge(df_exp_aligned, df_ai_grouped, on='问题', how='left', suffixes=('_exp', '_ai'))
                            else:
                                # Normal case: assume rows aligned or use '问题' to be safe?
                                # Previously we assumed row alignment.
                                # Let's stick to row alignment for matching files (147 rows)
                                # but use merge for 2-3-2 logic above.
                                # Wait, mixing logic is dangerous.
                                # If rows aligned, use direct index.
                                df_merged = pd.concat([df_exp_aligned.add_suffix('_exp'), df_ai.add_suffix('_ai')], axis=1)

                        else:
                            # Fallback: align by index blindly
                            df_merged = pd.concat([df_exp_aligned.add_suffix('_exp'), df_ai.add_suffix('_ai')], axis=1)

                        columns_ai = df_merged.columns[df_merged.columns.str.endswith('_ai')]
                        columns_exp = df_merged.columns[df_merged.columns.str.endswith('_exp')]

                        for m in metrics:
                             try:
                                 col_ai = f"{m}_ai"
                                 col_exp = f"{m}_exp"
                                 
                                 if col_ai in df_merged.columns and col_exp in df_merged.columns:
                                     val_a = df_merged[col_ai].fillna(0).round().astype(int)
                                     val_b = df_merged[col_exp].fillna(0).round().astype(int)
                                     
                                     if len(val_a) == len(val_b) and len(val_a) > 0:
                                         kappa_metrics[m] = cohen_kappa_score(val_a, val_b, weights='quadratic')
                                     else:
                                         kappa_metrics[m] = np.nan
                                 else:
                                     kappa_metrics[m] = np.nan
                             except Exception as e:
                                 # print(f"Error metric {m}: {e}")
                                 kappa_metrics[m] = np.nan
                        
                        current_model_results[col_name] = kappa_metrics
                    
                    df_model = pd.DataFrame(current_model_results)
                    
                    # Add Average column for this model (across 3 experts)
                    df_model['Avg (All Experts)'] = df_model.mean(axis=1)
                    
                    # Add Average row (across metrics)
                    # We want the average of the 'Avg (All Experts)' column, which is the overall model score
                    overall_avg = df_model['Avg (All Experts)'].mean()
                    
                    # Save to sheet
                    sheet_title = ai_name[:30]
                    df_model.to_excel(writer, sheet_name=sheet_title)
                    
                    # Parse data for Summary Sheet
                    # Get the 'Avg (All Experts)' col which contains per-metric averages
                    avg_col = df_model['Avg (All Experts)']
                    
                    row_data = {'Model': ai_name}
                    # Overall Average Kappa
                    row_data['Average Kappa'] = overall_avg
                    
                    # Per metric
                    for m in metrics:
                        if m in avg_col.index:
                            row_data[m] = avg_col.loc[m]
                    
                    summary_data.append(row_data)

                except Exception as e:
                    print(f"Error processing {ai_name}: {e}")

            # 3. Write Summary Sheet
            df_summary = pd.DataFrame(summary_data)
            # Reorder columns for readability
            desired_cols = ['Model', 'Average Kappa'] + metrics
            # Only keep cols that exist in df
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
