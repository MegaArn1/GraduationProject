
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr

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
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < len(metrics):
        return pd.Series([np.nan] * len(df))
    return df[metrics].fillna(0).astype(int).sum(axis=1)

def calculate_icc_proxy(y_true, y_pred):
    """
    Calculate a simple proxy for ICC (Intraclass Correlation Coefficient) type consistency
    using Variance analysis, as standard ICC requires specific ANOVA setup libraries.
    Here we calculate Concordance Correlation Coefficient (CCC) which is very close to ICC
    for assessing agreement.
    CCC = (2 * rho * sigma_x * sigma_y) / (sigma_x^2 + sigma_y^2 + (mu_x - mu_y)^2)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        return np.nan
        
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    # Pearson Correlation
    if sd_true == 0 or sd_pred == 0:
        rho = 0
    else:
        rho = np.corrcoef(y_true, y_pred)[0,1]
        
    # CCC Calculation
    numerator = 2 * rho * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    if denominator == 0:
        return 0
        
    ccc = numerator / denominator
    return ccc

def calculate_icc_manual(y1, y2):
    """
    Calculate ICC(2,1) (Absolute Agreement) and ICC(3,1) (Consistency) for 2 raters manually.
    Based on Shrout & Fleiss (1979) and McGraw & Wong (1996) formulas using Mean Squares.
    
    Arguments:
    y1, y2: Arrays of ratings from Rater 1 and Rater 2.
    
    Returns:
    (icc2_1, icc3_1)
    """
    import numpy as np
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    # Check lengths
    if len(y1) != len(y2) or len(y1) < 2:
        return np.nan, np.nan
        
    n = len(y1)
    k = 2  # Two raters
    
    # Create the matrix (n rows, 2 columns)
    # This is equivalent to a Two-Way ANOVA without replication (since 1 observation per rater per subject)
    
    # Calculate Mean Squares Manually
    # SStotal = var(all) * (N*k - 1) 
    # But let's build components:
    
    # Grand Mean
    bar_y = (np.sum(y1) + np.sum(y2)) / (2*n)
    
    # SStotal (Sum of squared differences from grand mean)
    sst = np.sum((y1 - bar_y)**2) + np.sum((y2 - bar_y)**2)
    
    # SSR (Sum of Squares for Rows/Subjects) -> Between-subject variability
    # Mean of each subject across raters
    subj_means = (y1 + y2) / 2
    ssr = k * np.sum((subj_means - bar_y)**2)
    
    # SSC (Sum of Squares for Columns/Raters) -> Between-rater variability
    # Mean of each rater across subjects
    rater_means = np.array([np.mean(y1), np.mean(y2)])
    ssc = n * np.sum((rater_means - bar_y)**2)
    
    # SSE (Sum of Squares for Error/Residual)
    sse = sst - ssr - ssc
    
    # Degrees of Freedom
    df_r = n - 1
    df_c = k - 1
    df_e = df_r * df_c
    
    # Mean Squares
    msr = ssr / df_r
    msc = ssc / df_c
    mse = sse / df_e
    
    # -------------------------------------------------------------------------
    # ICC(3,1) - Consistency
    # Definition: The correlation between any two measurements made on the same subject (fixed rater effect excluded)
    # Formula: (MSR - MSE) / (MSR + (k-1)*MSE)
    # -------------------------------------------------------------------------
    if (msr + (k-1)*mse) == 0:
         icc3 = 0
    else:
         icc3 = (msr - mse) / (msr + (k-1)*mse)
         
    # -------------------------------------------------------------------------
    # ICC(2,1) - Absolute Agreement
    # Definition: Agreement between two random raters (includes rater systematic error)
    # Formula: (MSR - MSE) / (MSR + (k-1)*MSE + (k/n)*(MSC - MSE))
    # -------------------------------------------------------------------------
    denom_icc2 = msr + (k-1)*mse + (k/n)*(msc - mse)
    
    if denom_icc2 == 0:
        icc2 = 0
    else:
        icc2 = (msr - mse) / denom_icc2
        
    return icc2, icc3

def main():
    if not os.path.exists(ai_file_path):
        print(f"Error: AI file {ai_file_path} not found.")
        return

    # 1. Load Expert Scores (Total Score)
    print("Loading Expert Scores...")
    experts_scores = {} 
    
    for name, path in expert_files.items():
        if os.path.exists(path):
            df = pd.read_excel(path)
            df.columns = df.columns.astype(str).str.strip()
            experts_scores[name] = calculate_total_score(df)
        else:
            print(f"Warning: Expert file {path} not found.")

    summary_rows = []
    
    # --- 2. Human Benchmark (Expert vs Expert) ---
    print("Calculating Human Benchmark (Metrics)...")
    expert_names = list(experts_scores.keys())
    human_stats = {'Pearson': [], 'Spearman': [], 'CCC': [], 'ICC2': [], 'ICC3': []}
    
    for i in range(len(expert_names)):
        for j in range(i + 1, len(expert_names)):
            name_a = expert_names[i]
            name_b = expert_names[j]
            
            score_a = experts_scores[name_a]
            score_b = experts_scores[name_b]
            
            mask = ~pd.isna(score_a) & ~pd.isna(score_b)
            if mask.sum() > 1:
                sa = score_a[mask]
                sb = score_b[mask]
                
                # Correlation
                p_corr, _ = pearsonr(sa, sb)
                s_corr, _ = spearmanr(sa, sb)
                
                # CCC (proxy for ICC)
                ccc = calculate_icc_proxy(sa, sb)
                
                # ICC(2) & ICC(3)
                icc2, icc3 = calculate_icc_manual(sa, sb)
                
                human_stats['Pearson'].append(p_corr)
                human_stats['Spearman'].append(s_corr)
                human_stats['CCC'].append(ccc)
                human_stats['ICC2'].append(icc2)
                human_stats['ICC3'].append(icc3)
    
    summary_rows.append({
        'Model': 'Human Experts (Ref)',
        'Pearson Correlation (r)': np.nanmean(human_stats['Pearson']),
        'Spearman Rank (rho)': np.nanmean(human_stats['Spearman']),
        'Concordance (CCC)': np.nanmean(human_stats['CCC']),
        'ICC(2,1) Abs Agree': np.nanmean(human_stats['ICC2']),
        'ICC(3,1) Consistency': np.nanmean(human_stats['ICC3'])
    })
    
    # --- 3. Process AI Models ---
    xls = pd.ExcelFile(ai_file_path)
    sheet_names = xls.sheet_names
    
    for sheet_name in sheet_names:
        print(f"Processing model: {sheet_name}...")
        try:
            df_ai = pd.read_excel(xls, sheet_name=sheet_name)
            df_ai.columns = df_ai.columns.astype(str).str.strip()
            ai_score = calculate_total_score(df_ai)
            
            if ai_score.isna().all():
                continue
                
            model_stats = {'Pearson': [], 'Spearman': [], 'CCC': [], 'ICC2': [], 'ICC3': []}
            
            for exp_name, exp_score in experts_scores.items():
                min_len = min(len(ai_score), len(exp_score))
                s_ai = ai_score.iloc[:min_len]
                s_exp = exp_score.iloc[:min_len]
                
                mask = ~pd.isna(s_ai) & ~pd.isna(s_exp)
                if mask.sum() > 1:
                    sa = s_ai[mask]
                    se = s_exp[mask]
                    
                    p_corr, _ = pearsonr(sa, se)
                    s_corr, _ = spearmanr(sa, se)
                    ccc = calculate_icc_proxy(sa, se)
                    icc2, icc3 = calculate_icc_manual(sa, se)
                    
                    model_stats['Pearson'].append(p_corr)
                    model_stats['Spearman'].append(s_corr)
                    model_stats['CCC'].append(ccc)
                    model_stats['ICC2'].append(icc2)
                    model_stats['ICC3'].append(icc3)
            
            summary_rows.append({
                'Model': sheet_name,
                'Pearson Correlation (r)': np.nanmean(model_stats['Pearson']),
                'Spearman Rank (rho)': np.nanmean(model_stats['Spearman']),
                'Concordance (CCC)': np.nanmean(model_stats['CCC']),
                'ICC(2,1) Abs Agree': np.nanmean(model_stats['ICC2']),
                'ICC(3,1) Consistency': np.nanmean(model_stats['ICC3'])
            })
            
        except Exception as e:
            print(f"Error processing {sheet_name}: {e}")
            
    # Output
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values('ICC(2,1) Abs Agree', ascending=False)
    
    output_file = 'kappa_scores_continuous_icc.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
    print("\n--- Continuous Score Consistency Analysis (Mean across 3 experts) ---")
    print(df_summary)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
