# Day_7_humanization.py
# 抗体人源化程度分析最终版
#
# 数据来源：TheraSAbDab
# 分析目标：
# 1. 根据 Genetics 列构建人源化分组
# 2. 分析不同人源化程度抗体的重链、轻链序列特征
# 3. 使用 PCA 可视化
# 4. 使用 ANOVA 找出不同人源化类别之间差异较大的特征
# 5. 保存分析结果和图片
#
# 鼠源抗体 → 嵌合抗体 → 人源化抗体 → 全人源/基因工程人源抗体

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif


# =========================================================
# 1. 基本参数设置
# =========================================================

# 你的原始数据路径
INPUT_FILE = r"D:\Mary.yu\study\code\TheraSAbDab_SeqStruc_OnlineDownload.csv"

# 输出结果保存文件夹
OUTPUT_DIR = r"D:\Mary.yu\study\code\Day_7_humanization_result"

# Genetics 列名：这一列更适合做人源化分析
GENETICS_COL = "Genetics (Bispecifics delimited with semicolon)"

# 人源化分组标签
LABEL_COL = "Humanization_Group"

# 是否只分析 Whole mAb
# True：只分析 Whole mAb，可以减少 Fab、scFv、Fusion Protein 等结构差异干扰
# False：分析全部抗体
# Whole mAb:完整单克隆抗体
ONLY_WHOLE_MAB = True

# 每个类别至少保留多少条样本
MIN_CLASS_COUNT = 5

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体，防止图片中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 2. 生成人源化分组标签
# =========================================================

def get_humanization_group(x):
    """
    根据 Genetics 列，把抗体划分为人源化相关类别。
    常见类别：
    1. Murine：鼠源抗体，人源化程度最低
    2. Chimeric：嵌合抗体
    3. Humanised：人源化抗体
    4. Genetically human：全人源或基因工程人源抗体
    """

    if pd.isna(x):
        return "Unknown"

    s = str(x).strip().lower()

    if s in ["", "nan", "na", "n/a", "none", "unknown", "tbc"]:
        return "Unknown"

    # 有些双抗可能用分号分隔多个来源
    parts = [p.strip() for p in s.split(";") if p.strip()]

    labels = []

    for p in parts:

        # 避免 humanised 被误判为 genetically human
        if "genetically human" in p or "fully human" in p:
            labels.append("Genetically human")

        elif "humanised" in p or "humanized" in p:
            # 有些写法是 chimeric and/or humanised，不够明确
            if "chimeric" in p:
                labels.append("Chimeric/Humanised ambiguous")
            else:
                labels.append("Humanised")

        elif "chimeric" in p:
            labels.append("Chimeric")

        elif "murine" in p or "mouse" in p:
            labels.append("Murine")

        elif "canine" in p or "feline" in p or "llama" in p or "alpaca" in p:
            labels.append("Other species")

        else:
            labels.append("Other/Unknown")

    unique_labels = set(labels)

    # 如果只有一个明确标签，直接返回
    if len(unique_labels) == 1:
        return list(unique_labels)[0]

    # 如果一个抗体里混合了多个来源，先标记为 Mixed
    return "Mixed"


# =========================================================
# 3. 清洗蛋白质序列
# =========================================================

def clean_protein_sequence(seq):
    """
    清洗蛋白质序列，只保留 20 种标准氨基酸字母。
    """
    if pd.isna(seq):
        return ""
    seq = str(seq).upper().strip()
    # 只保留标准氨基酸
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)

    return seq


# =========================================================
# 4. 计算单条序列的理化特征
# =========================================================

def compute_one_sequence_features(seq):
    """
    输入一条蛋白序列，计算它的基础理化特征。
    """
    seq = clean_protein_sequence(seq)

    # 序列太短就跳过
    if len(seq) < 10:
        return None

    try:
        analysis = ProteinAnalysis(seq)

        features = {
            "Length": len(seq),
            "Molecular_Weight": analysis.molecular_weight(),
            "Isoelectric_Point": analysis.isoelectric_point(),
            "GRAVY": analysis.gravy(),
            "Aromaticity": analysis.aromaticity(),
            "Instability_Index": analysis.instability_index(),
        }

        # 20 种氨基酸组成比例
        aa_percent = analysis.amino_acids_percent

        # 兼容不同 Biopython 版本
        if callable(aa_percent):
            aa_percent = aa_percent()

        features.update(aa_percent)

        return features

    except Exception:
        return None


# =========================================================
# 5. 从 DataFrame 构建特征矩阵
# =========================================================

def build_feature_table(df, seq_col, chain_name):
    """
    从原始表格中提取某一条链的序列特征。
    seq_col:
    - HeavySequence：重链
    - LightSequence：轻链
    """

    feature_rows = []

    for _, row in df.iterrows():

        seq = row.get(seq_col)
        features = compute_one_sequence_features(seq)

        # 如果该条序列无效，跳过
        if features is None:
            continue

        # 保留抗体名称
        features["ID"] = row.get("Therapeutic", "Unknown")

        # 保留 Format，方便后面检查
        features["Format"] = row.get("Format", "Unknown")

        # 保留原始 Genetics 信息
        features["Genetics"] = row.get(GENETICS_COL, "Unknown")

        # 保留人源化分组标签
        features[LABEL_COL] = row.get(LABEL_COL, "Unknown")

        # 添加链类型
        features["Chain"] = chain_name

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


# =========================================================
# 6. PCA 可视化
# =========================================================

def plot_pca(df_features, chain_name):
    """
    对序列特征做 PCA 降维，并按人源化类别着色。
    """

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print(f"{chain_name} 数值特征不足，无法进行 PCA")
        return

    X = df_features[numeric_cols].fillna(df_features[numeric_cols].mean())

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 降到二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        LABEL_COL: df_features[LABEL_COL].values
    })

    plt.figure(figsize=(10, 8))

    categories = df_plot[LABEL_COL].unique()

    for cat in categories:
        sub = df_plot[df_plot[LABEL_COL] == cat]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            alpha=0.65,
            s=45,
            label=f"{cat} (n={len(sub)})"
        )

    pc1_ratio = pca.explained_variance_ratio_[0]
    pc2_ratio = pca.explained_variance_ratio_[1]

    plt.xlabel(f"PC1 ({pc1_ratio:.2%})")
    plt.ylabel(f"PC2 ({pc2_ratio:.2%})")
    plt.title(f"{chain_name} 人源化分组 PCA 可视化")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"pca_{chain_name}_humanization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n{chain_name} PCA 结果：")
    print(f"PC1 解释方差：{pc1_ratio:.2%}")
    print(f"PC2 解释方差：{pc2_ratio:.2%}")
    print(f"前两个主成分累计解释方差：{pc1_ratio + pc2_ratio:.2%}")


# =========================================================
# 7. ANOVA 差异特征分析
# =========================================================

def find_sensitive_features(df_features, chain_name, top_k=15):
    """
    使用 ANOVA F 检验，找出不同人源化类别之间差异较大的特征。
    F_Score 越大，说明该特征越能区分不同人源化类别。
    """

    label_counts = df_features[LABEL_COL].value_counts()
    valid_labels = label_counts[label_counts >= MIN_CLASS_COUNT].index.tolist()

    df_valid = df_features[df_features[LABEL_COL].isin(valid_labels)].copy()

    if len(valid_labels) < 2:
        print(f"{chain_name} 有效人源化类别少于 2 个，无法进行 ANOVA 分析")
        return pd.DataFrame()

    numeric_cols = df_valid.select_dtypes(include=[np.number]).columns.tolist()

    X = df_valid[numeric_cols].fillna(df_valid[numeric_cols].mean())
    y = df_valid[LABEL_COL].values

    f_scores, p_values = f_classif(X, y)

    result = pd.DataFrame({
        "Feature": numeric_cols,
        "F_Score": f_scores,
        "P_Value": p_values
    })

    result = result.dropna().sort_values("F_Score", ascending=False)

    print(f"\n{chain_name} 中不同人源化类别差异最大的 Top {top_k} 特征：")
    print(result.head(top_k))

    # 画柱状图
    top_features = result.head(top_k)

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["F_Score"])
    plt.gca().invert_yaxis()
    plt.xlabel("F Score")
    plt.title(f"{chain_name} 不同人源化类别差异最大的 Top {top_k} 特征")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"anova_top_features_{chain_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, f"anova_features_{chain_name}.csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return result


# =========================================================
# 8. 人源化趋势分析
# =========================================================

def find_trend_features(df_features, chain_name, top_k=15):
    """
    趋势分析：
    把人源化程度转成数值：
    Murine = 0
    Chimeric = 1
    Humanised = 2
    Genetically human = 3

    然后计算每个特征与人源化程度的 Spearman 相关性。
    相关系数绝对值越大，说明该特征越可能随着人源化程度变化。
    """

    score_map = {
        "Murine": 0,
        "Chimeric": 1,
        "Humanised": 2,
        "Genetically human": 3
    }

    df_valid = df_features[df_features[LABEL_COL].isin(score_map.keys())].copy()

    if len(df_valid) < 10:
        print(f"{chain_name} 有效样本太少，无法做人源化趋势分析")
        return pd.DataFrame()

    df_valid["Humanization_Score"] = df_valid[LABEL_COL].map(score_map)

    numeric_cols = df_valid.select_dtypes(include=[np.number]).columns.tolist()

    # 不把 Humanization_Score 自己当作特征
    if "Humanization_Score" in numeric_cols:
        numeric_cols.remove("Humanization_Score")

    result_rows = []

    for feature in numeric_cols:
        corr = df_valid[[feature, "Humanization_Score"]].corr(method="spearman").iloc[0, 1]

        result_rows.append({
            "Feature": feature,
            "Spearman_Correlation": corr,
            "Abs_Correlation": abs(corr)
        })

    result = pd.DataFrame(result_rows)
    result = result.dropna().sort_values("Abs_Correlation", ascending=False)

    print(f"\n{chain_name} 中随人源化程度变化最明显的 Top {top_k} 特征：")
    print(result.head(top_k))

    # 画图
    top_features = result.head(top_k)

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["Spearman_Correlation"])
    plt.gca().invert_yaxis()
    plt.xlabel("Spearman Correlation")
    plt.title(f"{chain_name} 与人源化程度相关性最高的 Top {top_k} 特征")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"trend_top_features_{chain_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, f"trend_features_{chain_name}.csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return result


# =========================================================
# 9. 单条链完整分析流程
# =========================================================

def analyze_chain(df, seq_col, chain_name):
    """
    对重链或轻链进行完整分析：
    1. 构建特征矩阵
    2. PCA 可视化
    3. ANOVA 差异特征分析
    4. 人源化趋势分析
    """

    print("\n" + "=" * 70)
    print(f"开始分析：{chain_name}")
    print("=" * 70)

    df_features = build_feature_table(df, seq_col=seq_col, chain_name=chain_name)

    print(f"{chain_name} 有效序列数量：{len(df_features)}")
    print(f"{chain_name} 特征数量：{df_features.shape[1]}")

    print(f"\n{chain_name} 人源化分组分布：")
    print(df_features[LABEL_COL].value_counts())

    # 保存特征矩阵
    feature_path = os.path.join(OUTPUT_DIR, f"{chain_name}_features.csv")
    df_features.to_csv(feature_path, index=False, encoding="utf-8-sig")

    # PCA
    plot_pca(df_features, chain_name)

    # ANOVA 分析
    anova_result = find_sensitive_features(df_features, chain_name)

    # 趋势分析
    trend_result = find_trend_features(df_features, chain_name)

    return df_features, anova_result, trend_result


# =========================================================
# 10. 主程序
# =========================================================

if __name__ == "__main__":

    # 读取原始数据
    df = pd.read_csv(INPUT_FILE)

    print(f"原始数据共有 {len(df)} 条记录")
    print("\n前 5 行数据：")
    print(df.head())

    print("\n数据列名：")
    print(df.columns.tolist())

    # 检查 Genetics 列是否存在
    if GENETICS_COL not in df.columns:
        raise ValueError(f"找不到列：{GENETICS_COL}，请检查 CSV 文件列名是否一致")

    # 根据 Genetics 列生成人源化分组
    df[LABEL_COL] = df[GENETICS_COL].apply(get_humanization_group)

    print("\n原始人源化分组统计：")
    print(df[LABEL_COL].value_counts())

    # 保存带有人源化标签的原始表
    labeled_path = os.path.join(OUTPUT_DIR, "TheraSAbDab_with_Humanization_Group.csv")
    df.to_csv(labeled_path, index=False, encoding="utf-8-sig")

    # 如果只分析 Whole mAb，则过滤
    if ONLY_WHOLE_MAB:
        df = df[df["Format"] == "Whole mAb"].copy()
        print("\n已开启 ONLY_WHOLE_MAB=True")
        print("当前只分析 Format 为 Whole mAb 的抗体")
        print(f"过滤后剩余 {len(df)} 条记录")

    # 只保留明确的人源化类别
    keep_labels = [
        "Murine",
        "Chimeric",
        "Humanised",
        "Genetically human"
    ]

    df = df[df[LABEL_COL].isin(keep_labels)].copy()

    print("\n用于最终分析的人源化分组统计：")
    print(df[LABEL_COL].value_counts())

    # 保存最终用于分析的数据
    final_data_path = os.path.join(OUTPUT_DIR, "final_data_for_humanization_analysis.csv")
    df.to_csv(final_data_path, index=False, encoding="utf-8-sig")

    # 分析重链
    heavy_features, heavy_anova, heavy_trend = analyze_chain(
        df,
        seq_col="HeavySequence",
        chain_name="Heavy_Chain"
    )

    # 分析轻链
    light_features, light_anova, light_trend = analyze_chain(
        df,
        seq_col="LightSequence",
        chain_name="Light_Chain"
    )

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

    print(f"\n所有结果已保存到文件夹：")
    print(OUTPUT_DIR)

    print("\n主要输出文件包括：")
    print("1. TheraSAbDab_with_Humanization_Group.csv")
    print("2. final_data_for_humanization_analysis.csv")
    print("3. Heavy_Chain_features.csv")
    print("4. Light_Chain_features.csv")
    print("5. anova_features_Heavy_Chain.csv")
    print("6. anova_features_Light_Chain.csv")
    print("7. trend_features_Heavy_Chain.csv")
    print("8. trend_features_Light_Chain.csv")
    print("9. PCA 图片")
    print("10. ANOVA Top 特征图片")
    print("11. 人源化趋势特征图片")



"""
============================================================
抗体人源化程度分析总结报告
============================================================

本代码基于 TheraSAbDab 数据库中的治疗性抗体序列数据，
对抗体人源化程度与序列理化特征之间的关系进行初步分析。

一、研究目的
------------------------------------------------------------
本实验希望分析不同人源化程度抗体之间是否存在可检测的
序列特征差异，并进一步筛选出可能与人源化程度相关的关键特征。

本实验按照 Genetics 字段将抗体划分为以下四类：
Murine              ：鼠源抗体，人源化程度最低
Chimeric            ：嵌合抗体
Humanised           ：人源化抗体
Genetically human   ：全人源 / 基因工程人源抗体，人源化程度最高

即人源化程度顺序为：
Murine → Chimeric → Humanised → Genetically human
鼠源抗体 → 嵌合抗体 → 人源化抗体 → 全人源抗体

二、数据筛选
------------------------------------------------------------
原始数据共有 1133 条记录。

为了减少不同抗体结构形式带来的干扰，本实验只保留 Format 为
Whole mAb 的完整单克隆抗体进行分析。筛选后剩余 800 条记录。

进一步只保留人源化标签明确的四类抗体：
Genetically human    357 条
Humanised            304 条
Chimeric              43 条
Murine                10 条

最终重链和轻链均获得 714 条有效序列用于分析。

三、特征提取
------------------------------------------------------------
本实验分别对重链 HeavySequence 和轻链 LightSequence 提取序列特征，
主要包括：

1. 序列长度 Length
2. 分子量 Molecular_Weight
3. 等电点 Isoelectric_Point
4. 疏水性 GRAVY
5. 芳香性 Aromaticity
6. 不稳定指数 Instability_Index
7. 20 种氨基酸组成比例，如 A、C、D、E、K、R 等

这些特征属于基础理化特征，可用于初步观察不同人源化类别之间
是否存在序列组成和性质上的差异。

四、PCA 分析结果
------------------------------------------------------------
PCA 分析用于将多维特征降到二维空间，方便观察不同人源化类别
在整体序列特征上的分布情况。

重链 PCA 结果：
PC1 解释方差为 15.16%
PC2 解释方差为 12.73%
前两个主成分累计解释方差为 27.88%

轻链 PCA 结果：
PC1 解释方差为 15.89%
PC2 解释方差为 12.75%
前两个主成分累计解释方差为 28.64%

结果说明：
基础理化特征能够反映不同人源化类别之间的一定差异，
但前两个主成分只能解释约 28% 的总体差异。
因此，PCA 图主要用于辅助观察，不能单独作为分类效果的证明。

五、ANOVA 差异特征分析结果
------------------------------------------------------------
ANOVA F 检验用于寻找不同人源化类别之间差异较大的特征。
F_Score 越大，说明该特征在不同类别之间的差异越明显。

重链中差异较明显的特征包括：
GRAVY、K、V、R、T、S、G、Length、Instability_Index、Y 等。

其中 GRAVY 和 K 的 F_Score 较高，说明重链疏水性和赖氨酸比例
在不同人源化类别之间存在较明显差异。

轻链中差异较明显的特征包括：
K、M、A、Molecular_Weight、H、R、Aromaticity、G、Q、S 等。

其中 K 在轻链中的 F_Score 最高，说明赖氨酸比例是区分
不同人源化类别的重要特征之一。

六、人源化趋势分析结果
------------------------------------------------------------
趋势分析将人源化程度转化为数值：
Murine = 0
Chimeric = 1
Humanised = 2
Genetically human = 3

然后计算各特征与人源化程度之间的 Spearman 相关性。

重链中，与人源化程度相关性较高的特征包括：
K、GRAVY、T、G、R、Y、Length、C、Instability_Index、V 等。

其中：
K 与人源化程度呈负相关，说明随着人源化程度升高，
重链中的赖氨酸比例整体下降。

GRAVY 与人源化程度呈正相关，说明随着人源化程度升高，
重链整体疏水性指标有上升趋势。

轻链中，与人源化程度相关性较高的特征包括：
K、H、Molecular_Weight、M、A、Aromaticity、R、Y、G 等。

其中：
K 与人源化程度呈负相关，说明轻链中的赖氨酸比例也随着
人源化程度升高而整体下降。

H 与人源化程度呈负相关，说明轻链中的组氨酸比例可能也与
人源化程度变化有关。

七、主要结论
------------------------------------------------------------
本实验结果表明，在 Whole mAb 数据集中，不同人源化程度抗体
在重链和轻链的基础理化特征上存在一定差异。

其中，赖氨酸 K 是最值得关注的特征之一。
K 在重链和轻链中都表现出较强的差异性，并且整体上随着
人源化程度升高而下降。

此外，重链 GRAVY、轻链 H、Aromaticity、Molecular_Weight 等特征
也可能与抗体人源化程度有关。

因此，本实验可以初步说明：
基础序列理化特征能够在一定程度上反映抗体人源化程度差异，
其中 K、GRAVY、H 等特征可能是重要的候选指标。

八、局限性
------------------------------------------------------------
1. 样本数量不平衡：
   Murine 样本只有 10 条，明显少于 Humanised 和 Genetically human，
   因此与鼠源抗体相关的结论需要谨慎解释。

2. 特征较基础：
   当前只使用 ProtParam 提取的基础理化特征，尚未加入更专业的
   人源化指标，例如人类 germline 相似度、FR 区相似度、
   CDR 区保留程度和免疫原性风险评分等。

3. PCA 解释率有限：
   前两个主成分累计解释方差约为 28%，说明二维 PCA 图不能完全
   表示全部序列特征差异。

4. 相关性不等于因果：
   ANOVA 和 Spearman 相关性只能说明特征与人源化类别之间存在统计关系，
   不能直接证明某个氨基酸变化导致人源化程度提高。

九、后续改进方向
------------------------------------------------------------
后续可以进一步加入：
1. 关键特征的箱线图，如 K、GRAVY、H 等；
2. 不同人源化类别之间的两两比较；
3. 多重检验校正，如 FDR 校正；
4. 与人类 germline 序列的相似度分析；
5. 框架区 FR 和互补决定区 CDR 的分区特征分析；
6. MHC-II 表位或免疫原性风险预测。

总体来看，本代码完成了一个基于 TheraSAbDab 数据的
抗体人源化程度初步序列特征分析流程，可作为后续深入研究的基础。
============================================================
"""