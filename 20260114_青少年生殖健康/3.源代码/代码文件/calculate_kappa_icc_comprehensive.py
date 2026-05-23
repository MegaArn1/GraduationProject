import pandas as pd
import numpy as np
import os
import itertools
from sklearn.metrics import cohen_kappa_score

# 评分维度
metrics = ['1. 指导性', '2. 准确性', '3. 完整性', '4. 安全性', '5. 易于理解', '6. 仅提供必要信息']

expert_files = {
    '专家A(丁波)': '专家审核表格（丁波-东南大学附属中大医院）.xlsx',
    '专家B(周定杰)': '专家审核表格（周定杰-江苏卫生健康发展研究中心）.xlsx',
    '专家C(孙玮)': '专家审核表格（孙玮-江苏省人民医院）.xlsx'
}

ai_file = '专家审核表格_1.7_scored_by_AI_async_2-3-2.xlsx'

def calculate_icc(data):
    """
    计算 ICC(2,1) 和 ICC(3,1)
    data: numpy array of shape (n_subjects, k_raters)
    """
    n, k = data.shape
    if k < 2 or n < 2:
        return np.nan, np.nan
        
    x_mean = np.mean(data)
    sst = np.sum((data - x_mean) ** 2)
    
    sr = np.sum((np.mean(data, axis=1) - x_mean) ** 2) * k
    sc = np.sum((np.mean(data, axis=0) - x_mean) ** 2) * n
    se = sst - sr - sc
    
    msr = sr / (n - 1) if n > 1 else 0
    msc = sc / (k - 1) if k > 1 else 0
    mse = se / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0
    
    # ICC(3,1): 趋势一致性 (Consistency)
    icc31 = (msr - mse) / (msr + (k - 1) * mse) if (msr + (k - 1) * mse) != 0 else np.nan
    
    # ICC(2,1): 绝对一致性 (Absolute Agreement)
    denom21 = msr + (k - 1) * mse + k * (msc - mse) / n
    icc21 = (msr - mse) / denom21 if denom21 != 0 else np.nan
    
    return icc31, icc21

def calculate_avg_kappa(data):
    """
    使用平均两两二次加权Kappa来衡量多个评价者之间的一致性
    data: numpy array of shape (n_subjects, k_raters)
    """
    n, k = data.shape
    if k < 2: return np.nan
    kappas = []
    for i in range(k):
        for j in range(i+1, k):
            try:
                kappas.append(cohen_kappa_score(data[:, i], data[:, j], weights='quadratic'))
            except:
                pass
    return np.mean(kappas) if kappas else np.nan

def clean_col(df):
    df.columns = df.columns.astype(str).str.strip()
    return df

def main():
    # 1. 载入数据
    # 专家数据
    experts_dfs = {}
    for name, path in expert_files.items():
        if os.path.exists(path):
            experts_dfs[name] = clean_col(pd.read_excel(path))
            
    # AI数据
    ai_raw = clean_col(pd.read_excel(ai_file))
    ai_models = ai_raw['model'].unique()
    ai_dfs = {model: ai_raw[ai_raw['model'] == model].reset_index(drop=True) for model in ai_models}
    
    # AI总体平均 (每个问题由5个AI打分的均值，四舍五入作为综合意见)
    # 取第一个AI df作为骨架，覆盖分数
    ai_overall = ai_dfs[ai_models[0]].copy()
    for m in metrics:
        all_scores = np.column_stack([ai_dfs[mod][m].fillna(0).astype(int) for mod in ai_models])
        ai_overall[m] = np.mean(all_scores, axis=1).round().astype(int)
    ai_dfs['AI总体'] = ai_overall
    
    # 专家总体平均
    expert_names = list(experts_dfs.keys())
    exp_overall = experts_dfs[expert_names[0]].copy()
    for m in metrics:
        all_scores = np.column_stack([experts_dfs[exp][m].fillna(0).astype(int) for exp in expert_names])
        exp_overall[m] = np.mean(all_scores, axis=1).round().astype(int)
    experts_dfs['专家总体'] = exp_overall
    
    # 创建Excel Writer
    output_file = '综合一致性分析表(Kappa与ICC)_1.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # --- Sheet 1: 表单说明 ---
        explanation_df = pd.DataFrame([
            {'表单名': 'AI_vs_AI', '评估对象': '计算不同AI模型之间的打分一致性', '如何计算': '针对各个维度(列)，收集5个AI在这147题上的各自打分，组建成 147行x5列 的矩阵。二次加权Kappa: 两两计算后再取平均。ICC(3,1): 单因素固定效应模型下的组内相关系数，反映模型间打分趋势一致性（是否同高同低）。ICC(2,1): 双因素随机效应模型的组内相关系数，反映模型间是否给出了完全一致的绝对分数。'},
            {'表单名': 'Expert_vs_Expert', '评估对象': '计算3位人类专家之间的打分一致性', '如何计算': '同上，矩阵大小为 147行x3列。'},
            {'表单名': 'AI_vs_Expert (Pairwise)', '评估对象': '每个AI与每个专家的1对1一致性', '如何计算': '147行x2列 的打分数据。直接计算两者的二次加权Kappa以及二元ICC。'},
            {'表单名': 'AI总体_vs_专家总体', '评估对象': 'AI平均分与专家平均分的一致性', '如何计算': 'AI总体为5个AI的打分平均(四舍五入取整), 专家总体为3个专家的打分平均。将这两组“共识”数据进行两两比对(1条AI平均，1条专家平均)。'}
        ])
        explanation_df.to_excel(writer, sheet_name='指引与说明', index=False)
        
        # 通用计算函数
        def generate_sheet_for_group(group_dfs, out_sheet_name, row_names):
            results = []
            for m in metrics:
                try:
                    data_matrix = np.column_stack([group_dfs[name][m].fillna(0).astype(int).values for name in row_names])
                    kappa = calculate_avg_kappa(data_matrix)
                    icc31, icc21 = calculate_icc(data_matrix)
                    results.append({'评价维度': m, '平均二次加权Kappa': kappa, 'ICC(3,1)趋势一致': icc31, 'ICC(2,1)绝对一致': icc21})
                except Exception as e:
                    pass
            pd.DataFrame(results).to_excel(writer, sheet_name=out_sheet_name, index=False)

        # --- Sheet 2: AI vs AI ---
        generate_sheet_for_group(ai_dfs, 'AI_vs_AI_多模型内部一致性', ai_models)
        
        # --- Sheet 3: Expert vs Expert ---
        generate_sheet_for_group(experts_dfs, 'Expert_vs_Expert_专家内部', expert_names)
        
        # --- Sheet 4: Pairwise AI vs Expert ---
        pw_results = []
        for ai_name in ai_models:
            for exp_name in expert_names:
                for m in metrics:
                    try:
                        ai_scores = ai_dfs[ai_name][m].fillna(0).astype(int).values
                        exp_scores = experts_dfs[exp_name][m].fillna(0).astype(int).values
                        data_matrix = np.column_stack([ai_scores, exp_scores])
                        
                        kappa = cohen_kappa_score(ai_scores, exp_scores, weights='quadratic')
                        icc31, icc21 = calculate_icc(data_matrix)
                        pw_results.append({
                            'AI模型': ai_name,
                            '人类专家': exp_name,
                            '维度': m,
                            '二次加权Kappa': kappa,
                            'ICC(3,1)': icc31,
                            'ICC(2,1)': icc21
                        })
                    except:
                        pass
        pd.DataFrame(pw_results).to_excel(writer, sheet_name='AI与专家1对1对比', index=False)
        
        # --- Sheet 5: AI总体 vs 专家总体 ---
        # AI总体 = "AI总体" ; 专家总体 = "专家总体"
        overall_results = []
        for m in metrics:
            try:
                ai_scores = ai_dfs['AI总体'][m].fillna(0).astype(int).values
                exp_scores = experts_dfs['专家总体'][m].fillna(0).astype(int).values
                data_matrix = np.column_stack([ai_scores, exp_scores])
                
                kappa = cohen_kappa_score(ai_scores, exp_scores, weights='quadratic')
                icc31, icc21 = calculate_icc(data_matrix)
                overall_results.append({
                    '评价维度': m,
                    '二次加权Kappa': kappa,
                    'ICC(3,1)': icc31,
                    'ICC(2,1)': icc21
                })
            except:
                pass
        pd.DataFrame(overall_results).to_excel(writer, sheet_name='AI总体_vs_专家总体', index=False)

    print(f"成功计算并导出至 {output_file}")

if __name__ == '__main__':
    main()