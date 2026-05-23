
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr

# Configuration
metrics = ['1. 指导性', '2. 准确性', '3. 完整性', '4. 安全性', '5. 易于理解', '6. 仅提供必要信息']

# File paths
ai_file_path = '专家审核表格_1.7_scored_by_AI_async_2-3.xlsx'

def calculate_total_score(df):
    """Calculate total score from the 6 metrics."""
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < len(metrics):
        return pd.Series([np.nan] * len(df))
    return df[metrics].fillna(0).astype(int).sum(axis=1)

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

    xls = pd.ExcelFile(ai_file_path)
    sheet_names = xls.sheet_names
    
    # Load all AI scores
    ai_scores = {}
    print("Loading AI Scores...")
    for sheet_name in sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = df.columns.astype(str).str.strip()
            score = calculate_total_score(df)
            if not score.isna().all():
                 ai_scores[sheet_name] = score
        except Exception as e:
            print(f"Error loading {sheet_name}: {e}")

    summary_rows = []
    
    # Compare each pair of AI models
    ai_names = list(ai_scores.keys())
    print(f"Comparing {len(ai_names)} AI models pairwise...")
    
    for i in range(len(ai_names)):
        for j in range(i + 1, len(ai_names)):
            name_a = ai_names[i]
            name_b = ai_names[j]
            
            score_a = ai_scores[name_a]
            score_b = ai_scores[name_b]
            
            # Align lengths
            min_len = min(len(score_a), len(score_b))
            sa = score_a.iloc[:min_len]
            sb = score_b.iloc[:min_len]
            
            mask = ~pd.isna(sa) & ~pd.isna(sb)
            
            if mask.sum() > 1:
                sa_valid = sa[mask]
                sb_valid = sb[mask]
                
                # Calculate metrics
                p_corr, _ = pearsonr(sa_valid, sb_valid)
                s_corr, _ = spearmanr(sa_valid, sb_valid)
                icc2, icc3 = calculate_icc_manual(sa_valid, sb_valid)
                
                summary_rows.append({
                    'Model A': name_a,
                    'Model B': name_b,
                    'Pearson (r)': p_corr,
                    'ICC(2,1) Abs Agree': icc2,
                    'ICC(3,1) Consistency': icc3
                })

    # Output
    df_summary = pd.DataFrame(summary_rows)
    # Sort by Consistency
    df_summary = df_summary.sort_values('ICC(3,1) Consistency', ascending=False)
    
    output_file = 'icc_scores_ai_vs_ai.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        df_summary.to_excel(writer, sheet_name='AI_vs_AI', index=False)
        
    print("\n--- AI vs AI Consistency Analysis (Top 10 Pairs) ---")
    print(df_summary.head(10))
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
