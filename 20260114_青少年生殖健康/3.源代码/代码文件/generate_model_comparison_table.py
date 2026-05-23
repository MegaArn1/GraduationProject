#!/usr/bin/env python3
"""
生成"各参评模型与临床专家打分的平均一致性对比"Markdown表格
数据来源：综合一致性分析表(Kappa与ICC)_8.xlsx -> "AI与专家1对1对比" sheet
"""

import pandas as pd


def format_model_name(full_name: str) -> str:
    """将完整的模型路径简化为论文中使用的名称"""
    name_map = {
        "Pro/zai-org/GLM-5": "GLM-5",
        "deepseek-ai/DeepSeek-V3.2": "DeepSeek-V3.2",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
        "Pro/Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    }
    return name_map.get(full_name, full_name)


def generate_markdown_table(
    excel_path: str = "综合一致性分析表(Kappa与ICC)_9.xlsx",
    sheet_name: str = "AI与专家1对1对比",
    output_path: str = None,
) -> str:
    """
    从Excel中读取AI与专家的1对1对比数据，计算各模型与两位专家的平均Kappa，
    并输出Markdown格式的三线表。
    
    Parameters
    ----------
    excel_path : str
        Excel文件路径
    sheet_name : str
        工作表名称
    output_path : str, optional
        若指定，则将Markdown表格写入该文件
    
    Returns
    -------
    str
        Markdown格式的表格字符串
    """
    # 1. 读取数据
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # 2. 简化模型名称
    df["模型简称"] = df["AI模型"].apply(format_model_name)
    
    # 3. 提取专家标识（专家A=周定杰，专家C=孙玮）
    df["专家标识"] = df["人类专家"].map({
        "专家A(周定杰)": "专家A(周定杰)",
        "专家C(孙玮)": "专家C(孙玮)",
    })
    
    # 4. 按模型+专家分组，计算六维度的平均Kappa
    grouped = df.groupby(["模型简称", "专家标识"])["二次加权Kappa"].mean().reset_index()
    
    # 5. 透视：行=模型，列=专家
    pivot = grouped.pivot(index="模型简称", columns="专家标识", values="二次加权Kappa")
    
    # 6. 确保两列都存在
    for col in ["专家A(周定杰)", "专家C(孙玮)"]:
        if col not in pivot.columns:
            pivot[col] = float('nan')
    
    # 7. 计算综合平均分用于排序（让表现好的排在后面，DeepSeek放最后突出）
    pivot["综合平均"] = pivot[["专家A(周定杰)", "专家C(孙玮)"]].mean(axis=1)
    pivot = pivot.sort_values("综合平均", ascending=True)
    
    # 8. 格式化数值（保留3位小数）
    for col in ["专家A(周定杰)", "专家C(孙玮)", "综合平均"]:
        pivot[col] = pivot[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    
    # 9. 构建Markdown表格
    lines = []
    lines.append("**表 5-X 各参评模型与临床专家打分的平均一致性对比**")
    lines.append("")
    lines.append("| 参评模型 | 与专家A(周定杰) 平均Kappa | 与专家C(孙玮) 平均Kappa |")
    lines.append("| :--- | :--- | :--- |")
    
    for model in pivot.index:
        kappa_a = pivot.loc[model, "专家A(周定杰)"]
        kappa_c = pivot.loc[model, "专家C(孙玮)"]
        
        # DeepSeek加粗突出
        if "DeepSeek" in model:
            lines.append(f"| **{model}** | **{kappa_a}** | **{kappa_c}** |")
        else:
            lines.append(f"| {model} | {kappa_a} | {kappa_c} |")
    
    markdown = "\n".join(lines)
    
    # 10. 若指定输出路径则写入文件
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Markdown表格已写入: {output_path}")
    
    return markdown


if __name__ == "__main__":
    # 使用示例
    md_table = generate_markdown_table(
        excel_path="../work/综合一致性分析表(Kappa与ICC)_9.xlsx",
        sheet_name="AI与专家1对1对比",
        # output_path="model_comparison_table.md",  # 取消注释可写入文件
    )
    print(md_table)
