#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制第六章（6.3 自动化评估结果）所需的全部图表。
依赖：pandas, numpy, matplotlib, rouge-score, jieba, scipy

运行前请确保已安装中文字体（如 SimHei / WenQuanYi Micro Hei），
否则中文标签可能显示为方块。
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from rouge_score import rouge_scorer
import jieba
import os

# ================= 配置区域 =================
# 若系统无 SimHei，可改为系统中实际存在的中文字体，如 'WenQuanYi Micro Hei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "."
GOLD_FILE = "bilibili_qa_dataset_3_scored.xlsx"
SCORED_DIR = "scored_base_and_finetune"

# ================= 数据读取 =================
def load_gold():
    df = pd.read_excel(GOLD_FILE)
    valid = df['is_valid_question'].astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})
    return df[(df['split'] == 'test') & valid][['id', 'ans_detail', 'Tag']].copy()


def load_model(path):
    df = pd.read_excel(path)
    valid = df['is_valid_question'].astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})
    return df[(df['split'] == 'test') & valid].copy()


# ================= 图 6-1: ROUGE-L 对比柱状图 =================
def plot_fig_rouge_comparison(gold_test):
    model_pairs = [
        ('qwen2.5-1.5b',
         'bilibili_qa_dataset_3_answered_qwen2.5-1.5b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-1.5b-instruct_finetuned_scored.xlsx'),
        ('qwen2.5-3b',
         'bilibili_qa_dataset_3_answered_qwen2.5-3b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-3b-instruct_finetuned_scored.xlsx'),
        ('qwen2.5-7b',
         'bilibili_qa_dataset_3_answered_qwen2.5-7b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-7b-instruct_finetuned_scored.xlsx'),
        ('qwen3-1.7b',
         'bilibili_qa_dataset_3_answered_qwen3-1.7b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-1.7b_finetuned_scored.xlsx'),
        ('qwen3-4b',
         'bilibili_qa_dataset_3_answered_qwen3-4b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-4b-instruct_finetuned_scored.xlsx'),
        ('qwen3-8b',
         'bilibili_qa_dataset_3_answered_qwen3-8b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-8b-instruct_finetuned_scored.xlsx'),
    ]

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    results = []

    for name, base_f, fin_f in model_pairs:
        base_df = load_model(os.path.join(SCORED_DIR, base_f))
        fin_df = load_model(os.path.join(SCORED_DIR, fin_f))
        merged_base = gold_test[['id', 'ans_detail']].merge(
            base_df[['id', 'ans_detail']], on='id', suffixes=('_gold', '_pred'))
        merged_fin = gold_test[['id', 'ans_detail']].merge(
            fin_df[['id', 'ans_detail']], on='id', suffixes=('_gold', '_pred'))

        def avg_rl(merged):
            scores = []
            for _, row in merged.iterrows():
                ref = ' '.join(jieba.cut(str(row['ans_detail_gold']).strip()))
                hyp = ' '.join(jieba.cut(str(row['ans_detail_pred']).strip()))
                if ref.strip() and hyp.strip():
                    scores.append(scorer.score(ref, hyp)['rougeL'].fmeasure)
            return np.mean(scores) if scores else 0

        base_mean = avg_rl(merged_base)
        fin_mean = avg_rl(merged_fin)

        # Wilcoxon
        base_scores = []
        fin_scores = []
        for _, row in merged_base.iterrows():
            ref = ' '.join(jieba.cut(str(row['ans_detail_gold']).strip()))
            hyp = ' '.join(jieba.cut(str(row['ans_detail_pred']).strip()))
            if ref.strip() and hyp.strip():
                base_scores.append(scorer.score(ref, hyp)['rougeL'].fmeasure)
        for _, row in merged_fin.iterrows():
            ref = ' '.join(jieba.cut(str(row['ans_detail_gold']).strip()))
            hyp = ' '.join(jieba.cut(str(row['ans_detail_pred']).strip()))
            if ref.strip() and hyp.strip():
                fin_scores.append(scorer.score(ref, hyp)['rougeL'].fmeasure)

        _, p = wilcoxon(base_scores, fin_scores, alternative='two-sided')
        results.append({'model': name, 'base': base_mean, 'fin': fin_mean, 'p': p})

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['base'], width, label='Base', color='#5B9BD5', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, df['fin'], width, label='Finetuned', color='#ED7D31', edgecolor='black', linewidth=0.5)

    for i, row in df.iterrows():
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
        max_h = max(row['base'], row['fin'])
        ax.text(i, max_h + 0.005, sig, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('ROUGE-L F1 Score', fontsize=12)
    ax.set_title('图 6-1  ROUGE-L Score Comparison: Base vs. Finetuned Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(df['fin'].max(), df['base'].max()) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_rouge_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_rouge_comparison.png")


# ================= 图 6-2: 综合均分对比柱状图 =================
def plot_fig_overall_score():
    model_pairs = [
        ('qwen2.5-1.5b',
         'bilibili_qa_dataset_3_answered_qwen2.5-1.5b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-1.5b-instruct_finetuned_scored.xlsx'),
        ('qwen2.5-3b',
         'bilibili_qa_dataset_3_answered_qwen2.5-3b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-3b-instruct_finetuned_scored.xlsx'),
        ('qwen2.5-7b',
         'bilibili_qa_dataset_3_answered_qwen2.5-7b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-7b-instruct_finetuned_scored.xlsx'),
        ('qwen3-1.7b',
         'bilibili_qa_dataset_3_answered_qwen3-1.7b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-1.7b_finetuned_scored.xlsx'),
        ('qwen3-4b',
         'bilibili_qa_dataset_3_answered_qwen3-4b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-4b-instruct_finetuned_scored.xlsx'),
        ('qwen3-8b',
         'bilibili_qa_dataset_3_answered_qwen3-8b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-8b-instruct_finetuned_scored.xlsx'),
    ]

    score_cols = ['guidance_score', 'accuracy_score', 'completeness_score',
                  'safety_score', 'understandability_score', 'conciseness_score']
    rows = []
    for name, base_f, fin_f in model_pairs:
        base_df = load_model(os.path.join(SCORED_DIR, base_f))
        fin_df = load_model(os.path.join(SCORED_DIR, fin_f))
        base_mean = base_df[score_cols].mean().mean()
        fin_mean = fin_df[score_cols].mean().mean()
        rows.append({'model': name, 'base': base_mean, 'fin': fin_mean})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    bars1 = ax.bar(x - width/2, df['base'], width, label='Base', color='#5B9BD5', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, df['fin'], width, label='Finetuned', color='#ED7D31', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Overall Mean Score (6 Dimensions)', fontsize=12)
    ax.set_title('图 6-2  Overall Quality Score: Base vs. Finetuned Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(1.5, 3.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for i, row in df.iterrows():
        ax.text(i - width/2, row['base'] + 0.03, f"{row['base']:.2f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, row['fin'] + 0.03, f"{row['fin']:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_overall_score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_overall_score_comparison.png")


# ================= 图 6-3: 六维度差异热力图 =================
def plot_fig_heatmap():
    model_pairs = [
        ('qwen2.5-1.5b',
         'bilibili_qa_dataset_3_answered_qwen2.5-1.5b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-1.5b-instruct_finetuned_scored.xlsx'),
        ('qwen2.5-3b',
         'bilibili_qa_dataset_3_answered_qwen2.5-3b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-3b-instruct_finetuned_scored.xlsx'),
        ('qwen2.5-7b',
         'bilibili_qa_dataset_3_answered_qwen2.5-7b-instruct_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen2.5-7b-instruct_finetuned_scored.xlsx'),
        ('qwen3-1.7b',
         'bilibili_qa_dataset_3_answered_qwen3-1.7b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-1.7b_finetuned_scored.xlsx'),
        ('qwen3-4b',
         'bilibili_qa_dataset_3_answered_qwen3-4b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-4b-instruct_finetuned_scored.xlsx'),
        ('qwen3-8b',
         'bilibili_qa_dataset_3_answered_qwen3-8b_base_scored.xlsx',
         'bilibili_qa_dataset_3_answered_qwen3-8b-instruct_finetuned_scored.xlsx'),
    ]

    score_cols = ['guidance_score', 'accuracy_score', 'completeness_score',
                  'safety_score', 'understandability_score', 'conciseness_score']
    dim_names = ['指导性', '准确性', '完整性', '安全性', '易于理解', '简洁性']
    dim_map = dict(zip(score_cols, dim_names))

    rows = []
    for name, base_f, fin_f in model_pairs:
        base_df = load_model(os.path.join(SCORED_DIR, base_f))
        fin_df = load_model(os.path.join(SCORED_DIR, fin_f))
        merged = base_df[['id'] + score_cols].merge(
            fin_df[['id'] + score_cols], on='id', suffixes=('_base', '_fin'))
        for col in score_cols:
            base_vals = merged[f"{col}_base"].dropna().values
            fin_vals = merged[f"{col}_fin"].dropna().values
            if len(base_vals) > 0:
                _, p = wilcoxon(base_vals, fin_vals, alternative='two-sided')
                rows.append({
                    'model': name,
                    'dimension': dim_map[col],
                    'mean_diff': np.mean(fin_vals) - np.mean(base_vals),
                    'p': p
                })

    ai_df = pd.DataFrame(rows)
    pivot = ai_df.pivot(index='model', columns='dimension', values='mean_diff')
    pivot = pivot[dim_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)

    ax.set_xticks(np.arange(len(dim_names)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(dim_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(dim_names)):
            model = pivot.index[i]
            dim = dim_names[j]
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue
            p_row = ai_df[(ai_df['model'] == model) & (ai_df['dimension'] == dim)]
            p = p_row['p'].values[0] if len(p_row) > 0 else 1.0
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            text = f"{val:+.2f}\n{sig}"
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if abs(val) > 0.5 else 'black', fontweight='bold')

    ax.set_title('图 6-3  Mean Score Difference (Finetuned - Base) across Six Dimensions',
                 fontsize=13, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Score Difference', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_ai_score_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_ai_score_heatmap.png")


# ================= 图 6-4: 主题级别提升横向条形图 =================
def plot_fig_theme_level(gold_test):
    base_path = os.path.join(SCORED_DIR, 'bilibili_qa_dataset_3_answered_qwen3-4b_base_scored.xlsx')
    fin_path = os.path.join(SCORED_DIR, 'bilibili_qa_dataset_3_answered_qwen3-4b-instruct_finetuned_scored.xlsx')

    base_df = load_model(base_path)
    fin_df = load_model(fin_path)

    score_cols = ['guidance_score', 'accuracy_score', 'completeness_score',
                  'safety_score', 'understandability_score', 'conciseness_score']

    base_df = base_df[['id'] + score_cols].merge(gold_test[['id', 'Tag']], on='id', how='left')
    fin_df = fin_df[['id'] + score_cols].merge(gold_test[['id', 'Tag']], on='id', how='left')
    merged = base_df.merge(fin_df, on='id', suffixes=('_base', '_fin'))

    theme_results = []
    for tag in sorted(merged['Tag'].dropna().unique()):
        sub = merged[merged['Tag'] == tag]
        base_scores = sub[[f"{c}_base" for c in score_cols]].mean(axis=1).dropna().values
        fin_scores = sub[[f"{c}_fin" for c in score_cols]].mean(axis=1).dropna().values
        if len(base_scores) == len(fin_scores) and len(base_scores) >= 5:
            _, p = wilcoxon(base_scores, fin_scores, alternative='two-sided')
            diff = np.mean(fin_scores) - np.mean(base_scores)
            theme_results.append({'Tag': tag, 'diff': diff, 'p': p})

    theme_df = pd.DataFrame(theme_results).sort_values('diff', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ED7D31' if d > 0 else '#5B9BD5' for d in theme_df['diff']]
    ax.barh(theme_df['Tag'], theme_df['diff'], color=colors, edgecolor='black', linewidth=0.5)

    for i, row in theme_df.iterrows():
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        ax.text(row['diff'] + 0.01 if row['diff'] > 0 else row['diff'] - 0.01, i, sig,
                ha='left' if row['diff'] > 0 else 'right', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Mean Score Difference (Finetuned - Base)', fontsize=12)
    ax.set_title('图 6-4  Theme-Level Performance Improvement (Qwen3-4B)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_theme_level_improvement.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_theme_level_improvement.png")


# ================= 主程序 =================
if __name__ == "__main__":
    gold_test = load_gold()
    print(f"Loaded {len(gold_test)} gold test samples.")

    plot_fig_rouge_comparison(gold_test)
    plot_fig_overall_score()
    plot_fig_heatmap()
    plot_fig_theme_level(gold_test)

    print("\nAll figures generated successfully.")
