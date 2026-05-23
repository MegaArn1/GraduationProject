
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

def calculate_total_score(df):
    """Calculate total score from the 6 metrics."""
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) < len(metrics):
        # Missing metrics
        return pd.Series([np.nan] * len(df))
    
    # Fill NaN with 0 before summing
    return df[metrics].fillna(0).astype(int).sum(axis=1)

def classify_quality(score):
    """
    Classify total score into 3 categories:
    Low Quality: 0-17 (Mapped to 0)
    Average: 18-23    (Mapped to 1)
    High Quality: 24-30 (Mapped to 2)
    """
    if pd.isna(score):
        return np.nan
    
    if score >= 24:
        return 2 # High
    elif score >= 18:
        return 1 # Average
    else:
        return 0 # Low

def calculate_categorical_stats(val_a, val_b):
    """Calculate QWK, Accuracy, and Random Agreement (Pe) for categorical data."""
    val_a = np.array(val_a)
    val_b = np.array(val_b)
    
    res = {}
    try:
        # Check alignment
        if len(val_a) != len(val_b):
             raise ValueError("Mismatch length")
             
        # Quadratic Weighted Kappa
        res['QWK'] = cohen_kappa_score(val_a, val_b, weights='quadratic')
        
        # Calculate Expected Agreement (Random Agreement Rate - Pe)
        # For simple unweighted Kappa, Pe = sum(pi_a * pi_b)
        # For weighted Kappa, the calculation of Pe is more complex involving the weights matrix.
        # But to explain "Random Agreement" simply to the user, the unweighted Pe is often what is meant conceptually 
        # (how often they agree by luck based on distribution). 
        # However, since we report Weighted Kappa, we should probably stick to the conceptual explanation or 
        # calculate the unweighted Pe as a proxy for "what if valid randomly".
        
        # Let's calculate simple (unweighted) Pe for demonstration
        # Get unique categories (0, 1, 2)
        cats = [0, 1, 2]
        pe_sum = 0
        n = len(val_a)
        if n > 0:
            for c in cats:
                p_a = np.sum(val_a == c) / n
                p_b = np.sum(val_b == c) / n
                pe_sum += (p_a * p_b)
        res['Random_Agreement_Pe'] = pe_sum
        
    except:
        res['QWK'] = np.nan
        res['Random_Agreement_Pe'] = np.nan
        
    try:
        res['Accuracy'] = accuracy_score(val_a, val_b)
    except:
        res['Accuracy'] = np.nan
        
    return res

def main():
    if not os.path.exists(ai_file_path):
        print(f"Error: AI file {ai_file_path} not found.")
        return

    # 1. Process Experts
    print("Processing Experts Categories...")
    experts_categories = {} 
    
    for name, path in expert_files.items():
        if os.path.exists(path):
            df = pd.read_excel(path)
            df.columns = df.columns.astype(str).str.strip()
            
            # Calculate Total Score
            totals = calculate_total_score(df)
            
            # Use map/apply
            cats = totals.apply(classify_quality)
            
            experts_categories[name] = cats
        else:
            print(f"Warning: Expert file {path} not found.")

    summary_rows = []
    output_file = 'kappa_scores_categorical_v2.xlsx'
    
    try:
        # --- 2. Human Benchmark (Expert vs Expert) ---
        print("Calculating Human Benchmark (Categorical)...")
        expert_names = list(experts_categories.keys())
        human_bench_stats = {'QWK': [], 'Accuracy': [], 'Pe': []}
        
        for i in range(len(expert_names)):
            for j in range(i + 1, len(expert_names)):
                name_a = expert_names[i]
                name_b = expert_names[j]
                
                cat_a = experts_categories[name_a]
                cat_b = experts_categories[name_b]
                
                # Filtering valid rows
                mask = ~pd.isna(cat_a) & ~pd.isna(cat_b)
                if mask.sum() > 0:
                    stats = calculate_categorical_stats(cat_a[mask], cat_b[mask])
                    human_bench_stats['QWK'].append(stats['QWK'])
                    human_bench_stats['Accuracy'].append(stats['Accuracy'])
                    human_bench_stats['Pe'].append(stats['Random_Agreement_Pe'])
        
        # Add Human Row
        summary_rows.append({
            'Model': 'Human Experts (Ref)',
            'Categorical QWK': np.nanmean(human_bench_stats['QWK']),
            'Classification Accuracy (Observed)': np.nanmean(human_bench_stats['Accuracy']),
            'Random Agreement (Expected)': np.nanmean(human_bench_stats['Pe'])
        })
        
        # --- 3. Process AI Models ---
        xls = pd.ExcelFile(ai_file_path)
        sheet_names = xls.sheet_names
        
        for sheet_name in sheet_names:
            print(f"Processing model: {sheet_name}...")
            try:
                df_ai = pd.read_excel(xls, sheet_name=sheet_name)
                df_ai.columns = df_ai.columns.astype(str).str.strip()
                
                # Calculate Total Score first
                ai_totals = calculate_total_score(df_ai)
                
                # Map to Cats
                ai_cats = ai_totals.apply(classify_quality)
                
                # Check for all NaNs (missing metrics)
                if ai_cats.isna().all():
                    print(f"  Skipping {sheet_name} (Missing metrics/NaNs)")
                    continue
                    
                model_stats = {'QWK': [], 'Accuracy': [], 'Pe': []}
                
                # Compare vs Each Expert
                for exp_name, exp_cats in experts_categories.items():
                    # Align lengths
                    length = min(len(ai_cats), len(exp_cats))
                    c_ai = ai_cats.iloc[:length]
                    c_exp = exp_cats.iloc[:length]
                    
                    mask = ~pd.isna(c_ai) & ~pd.isna(c_exp)
                    if mask.sum() > 0:
                        stats = calculate_categorical_stats(c_ai[mask], c_exp[mask])
                        model_stats['QWK'].append(stats['QWK'])
                        model_stats['Accuracy'].append(stats['Accuracy'])
                        model_stats['Pe'].append(stats['Random_Agreement_Pe'])
                
                # Add Model Row (Mean across 3 experts)
                summary_rows.append({
                    'Model': sheet_name,
                    'Categorical QWK': np.nanmean(model_stats['QWK']),
                    'Classification Accuracy (Observed)': np.nanmean(model_stats['Accuracy']),
                    'Random Agreement (Expected)': np.nanmean(model_stats['Pe'])
                })
                
            except Exception as e:
                print(f"Error processing {sheet_name}: {e}")
                
        # --- 4. Report and Save ---
        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values('Categorical QWK', ascending=False)
        
        print("\n--- Categorical Analysis Results (High/Avg/Low) ---")
        print(df_summary)

        with pd.ExcelWriter(output_file) as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        print(f"\nSaved categorical analysis to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    except Exception as e:
        print(f"Error writing excel: {e}")

if __name__ == "__main__":
    main()
