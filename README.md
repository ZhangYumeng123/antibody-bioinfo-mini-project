# antibody-bioinfo-mini-project
抗体序列生物信息学分析项目 | 入门
# Antibody Bioinformatics Mini Project

**生信入门项目 | 抗体序列分析 Pipeline**

## 📖 项目简介

本项目是我在实习期间利用业余时间完成的生信入门训练。目标是从零开始，用 Python 工具链完成抗体序列的读取、特征提取、数据库搜索和人源化程度分析。

**核心能力展示：**
- 生物序列数据处理（Biopython, SeqIO, ProtParam）
- 公共数据库 API 调用（NCBI Entrez）
- 数据可视化（Matplotlib, Seaborn, PCA）
- 可重复研究实践（environment.yml, Git 版本控制）

## 📂 项目结构

├── README.md
├── environment.yml
├── data/
│   └── example_sequences.fasta
├── notebooks/
│   ├── 01_sequence_loading_and_features.ipynb
│   ├── 02_ncbi_entrez_search.ipynb
│   └── 03_humanization_analysis.ipynb
└── scripts/
    └── antibody_pipeline.py

## 🚀 快速开始

### 1. 克隆仓库
git clone https://github.com/ZhangYumeng123/antibody-bioinfo-mini-project.git
cd antibody-bioinfo-mini-project

### 2. 创建环境
conda env create -f environment.yml
conda activate bioinfo

### 3. 运行 Notebook
jupyter notebook

## 🛠 技术栈

| 工具 | 用途 |
|------|------|
| Python 3.11 | 编程语言 |
| Biopython | 生物序列解析、ProtParam特征提取 |
| Pandas | 数据清洗与结构化 |
| Matplotlib / Seaborn | 热图、PCA可视化 |
| Scikit-learn | PCA降维、特征选择 |
| NCBI Entrez API | 公共数据库搜索同源序列 |
| Conda | 环境管理 |
| Git / GitHub | 版本控制与项目展示 |

## 📊 核心成果

1. 序列特征矩阵：将抗体序列转化为数值特征
2. 同源序列搜索：通过 NCBI Entrez API 自动搜索 Top5 相似序列
3. 人源化程度分析：PCA 降维可视化不同来源抗体的特征聚类

## 🙋 关于作者

- 生物信息学初学者，有后端开发实习经验
- 本项目是"将编程技能迁移至生物信息学领域"的实践记录
- GitHub: ZhangYumeng123

---
Last updated: 2026年7月
