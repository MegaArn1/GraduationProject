
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
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

def calculate_total_score(df):
    """Calculate total score from the 6 metrics."""
    # Ensure columns exist and fill with 0
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < len(metrics):
        # If vital columns missing, return NaN series
        return pd.Series([np.nan] * len(df))
    
    return df[metrics].fillna(0).astype(int).sum(axis=1)

def calculate_stats(val_a, val_b):
    """Calculate QWK, Pearson, Spearman, MAE for two integer arrays."""
    res = {}
    
    try:
        # 1. Quadratic Weighted Kappa (QWK)
        # Note: scikit-learn's implementation handles high cardinality integers fine for QWK
        res['QWK'] = cohen_kappa_score(val_a, val_b, weights='quadratic')
    except:
        res['QWK'] = np.nan
        
    try:
        # 2. Pearson Correlation
        corr, _ = pearsonr(val_a, val_b)
        res['Pearson_r'] = corr
    except:
        res['Pearson_r'] = np.nan

    try:
        # 3. Spearman Correlation
        corr, _ = spearmanr(val_a, val_b)
        res['Spearman_rho'] = corr
    except:
        res['Spearman_rho'] = np.nan
        
    try:
        # 4. Mean Absolute Error
        res['MAE'] = mean_absolute_error(val_a, val_b)
    except:
        res['MAE'] = np.nan
        
    try:
        # 5. Agreement within +/- 2 points (approx 10% of scale)
        diff = np.abs(val_a - val_b)
        res['Agreement_Within_2pts'] = np.mean(diff <= 2)
    except:
        res['Agreement_Within_2pts'] = np.nan
        
    return res

def main():
    if not os.path.exists(ai_file_path):
        print(f"Error: AI file {ai_file_path} not found.")
        return

    # Load Expert Data & Calculate Totals
    experts_data = {} # Stores (df, total_score_series)
    for name, path in expert_files.items():
        if os.path.exists(path):
            df = pd.read_excel(path)
            df.columns = df.columns.astype(str).str.strip()
            total_scores = calculate_total_score(df)
            experts_data[name] = total_scores
        else:
            print(f"Warning: Expert file {path} not found.")

    summary_rows = []
    output_file = 'kappa_scores_total_score.xlsx'
    
    try:
        with pd.ExcelWriter(output_file) as writer:
            
            # --- 1. Human Benchmark (Expert vs Expert) ---
            print("Calculating Human Benchmark (Total Score)...")
            expert_names = list(experts_data.keys())
            human_bench_results = {'QWK': [], 'Pearson_r': [], 'Spearman_rho': [], 'MAE': [], 'Agreement_Within_2pts': []}
            
            for i in range(len(expert_names)):
                for j in range(i + 1, len(expert_names)):
                    name_a = expert_names[i]
                    name_b = expert_names[j]
                    
                    scores_a = experts_data[name_a]
                    scores_b = experts_data[name_b]
                    
                    # Filter valid
                    mask = ~np.isnan(scores_a) & ~np.isnan(scores_b)
                    if mask.sum() > 0:
                        stats = calculate_stats(scores_a[mask], scores_b[mask])
                        for k, v in stats.items():
                            human_bench_results[k].append(v)
            
            # Add Human Row
            bench_row = {'Model': 'Human Experts (Ref)'}
            for k, v in human_bench_results.items():
                bench_row[k] = np.nanmean(v)
            summary_rows.append(bench_row)
            
            # --- 2. AI Models ---
            xls = pd.ExcelFile(ai_file_path)
            sheet_names = xls.sheet_names
            
            for sheet_name in sheet_names:
                print(f"Processing model: {sheet_name}...")
                try:
                    df_ai = pd.read_excel(xls, sheet_name=sheet_name)
                    df_ai.columns = df_ai.columns.astype(str).str.strip()
                    
                    ai_scores = calculate_total_score(df_ai)
                    
                    if ai_scores.isna().all():
                        print(f"  Skipping {sheet_name} due to missing metrics.")
                        continue
                        
                    # Compare with each expert
                    model_stats = {'QWK': [], 'Pearson_r': [], 'Spearman_rho': [], 'MAE': [], 'Agreement_Within_2pts': []}
                    
                    for exp_name, exp_scores in experts_data.items():
                        # Align lengths (truncate if mismatch, assuming sorted by Question)
                        # Or better, just use min length
                        min_len = min(len(ai_scores), len(exp_scores))
                        s_ai = ai_scores.iloc[:min_len]
                        s_exp = exp_scores.iloc[:min_len]
                        
                        mask = ~np.isnan(s_ai) & ~np.isnan(s_exp)
                        if mask.sum() > 0:
                            stats = calculate_stats(s_ai[mask], s_exp[mask])
                            for k, v in stats.items():
                                model_stats[k].append(v)
                    
                    # Add Model Row
                    row = {'Model': sheet_name}
                    for k, v in model_stats.items():
                        row[k] = np.nanmean(v)
                    summary_rows.append(row)
                    
                    # Also save detailed comparison for this model? Maybe not needed for now.
                    
                except Exception as e:
                    print(f"Error processing {sheet_name}: {e}")

            # --- 3. Save Summary ---
            df_summary = pd.DataFrame(summary_rows)
            # Reorder columns
            cols = ['Model', 'QWK', 'Pearson_r', 'Spearman_rho', 'MAE', 'Agreement_Within_2pts']
            df_summary = df_summary[cols]
            df_summary = df_summary.sort_values('QWK', ascending=False)
            
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print("\n--- Total Score Analysis Results ---")
            print(df_summary)
            
        print(f"\nSaved analysis to {output_file}")

    except Exception as e:
        print(f"Error writing excel: {e}")

if __name__ == "__main__":
    main()
