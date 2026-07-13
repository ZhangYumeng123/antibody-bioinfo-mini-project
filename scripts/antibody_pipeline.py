# 抗体序列分析 Pipeline
# 输入FASTA-> 特征矩阵 -> 相似性搜索 -> Top5输出

# 提问：什么是Top5输出？
# 就是你用 NCBI 搜索后，返回 E-value 最小（最相似）的前 5 条序列，输出它们的物种、长度、标题。
# retmax=5 就是 Top5

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import Entrez
import ssl
import time
import urllib.request
import urllib.error

# 提问：import 和from……import……区别?
# import pandas as pd                                  把整个pandas模块导入，用的时候加前缀 pd.DataFrame()
# from Bio import SeqIO                                只导入Bio模块里的SeqIO，直接用 SeqIO.parse()
# from Bio.SeqUtils.ProtParam import ProteinAnalysis   只导入这一个类，直接用 ProteinAnalysis()

# 解决SSL证书问题
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


def load_sequence(fasta_path):
    sequences = []  # ✅补充这一行
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        seq_id = record.id
        seq_length = len(seq)
        sequences.append({  # ✅每条序列装进字典
            "id": seq_id,
            "seq": seq,
        })
    return sequences  # ✅ 修正缩进：之前return在循环内部，现在移到外面


def extract_features(sequences):
    feature_list = []
    for item in sequences:
        seq_str = item["seq"]  # ✅循环每条序列，并从字典取序列字符串

        analysed_seq = ProteinAnalysis(seq_str)
        features = {
            "ID": item["id"],  # ✅修改：这里应该是序列ID，不是seq_str
            "Sequence": seq_str,  # ✅修改：这里才是序列本身
            "Length": len(seq_str),
            "Molecular_Weight": analysed_seq.molecular_weight(),
            "Isoelectric_Point": analysed_seq.isoelectric_point(),
            "GRAVY": analysed_seq.gravy(),
            "Aromaticity": analysed_seq.aromaticity(),
            "Instability_Index": analysed_seq.instability_index(),
        }
        aa_composition = analysed_seq.amino_acids_percent  # ✅用属性
        features.update(aa_composition)
        feature_list.append(features)
    return pd.DataFrame(feature_list)  # ✅返回 DataFrame，删掉 pass


def search_similar_sequence(query_seq, email, api_key=None, top_n=5, retries=3):
    """搜索单条序列的相似序列，带重试机制"""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # 设置较长的超时时间
    Entrez.timeout = 60

    # 使用序列的前30个氨基酸作为搜索词，并添加更通用的搜索条件
    search_term = f"{query_seq[:30]}[Sequence]"

    # 如果序列太短，使用完整序列
    if len(query_seq) < 20:
        search_term = f"{query_seq}[Sequence]"

    for attempt in range(retries):
        try:
            # 添加延迟，避免请求过于频繁
            if attempt > 0:
                time.sleep(2 ** attempt)  # 指数退避

            handle = Entrez.esearch(db="protein", term=search_term, retmax=top_n, idtype="acc")
            record = Entrez.read(handle)
            handle.close()

            id_list = record["IdList"]
            if not id_list:
                return []

            ids_str = ",".join(id_list)
            handle = Entrez.esummary(db="protein", id=ids_str)
            summaries = Entrez.read(handle)
            handle.close()

            results = []
            for summary in summaries:
                results.append({
                    "Accession": summary.get("AccessionVersion", "N/A"),
                    "Species": summary.get("Organism", "N/A"),
                    "Length": summary.get("Length", "N/A"),
                    "Title": summary.get("Title", "N/A"),
                    "Publish_Date": summary.get("CreateDate", "N/A")
                })
            return results

        except (urllib.error.URLError, ssl.SSLError, Exception) as e:
            print(f"搜索尝试 {attempt + 1}/{retries} 失败: {e}")
            if attempt == retries - 1:
                print(f"搜索最终失败: {e}")
                return []
            continue

    return []


def search_similar(df_features, email, api_key=None):
    """对特征矩阵中的所有序列进行相似性搜索"""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    Entrez.timeout = 60
    print("Entrez配置完成，准备搜索")

    all_results = []

    for idx, row in df_features.iterrows():
        seq_id = row["ID"]
        seq = row["Sequence"]
        print(f"正在搜索序列 {idx + 1}/{len(df_features)}: {seq_id}")

        # 使用整个序列进行搜索，但截取前100个氨基酸作为查询
        query_seq = seq[:100] if len(seq) > 100 else seq
        hits = search_similar_sequence(query_seq, email, api_key)

        if hits:
            for hit in hits:
                hit["Query_ID"] = seq_id
                all_results.append(hit)
            print(f"  ✅ 找到 {len(hits)} 条匹配结果")
        else:
            # 如果没有结果，记录空值
            print(f"  ❌ 未找到匹配结果")
            all_results.append({
                "Query_ID": seq_id,
                "Accession": "No hits",
                "Species": "No hits",
                "Length": "No hits",
                "Title": "No hits",
                "Publish_Date": "No hits"
            })

        # 添加延迟，避免请求过于频繁
        time.sleep(0.5)

    df = pd.DataFrame(all_results)
    print(f"\n找到 {len(df)} 条相似序列记录")
    return df


def run_pipeline(input_fasta, email, api_key=None):
    # 1.读取
    sequences = load_sequence(input_fasta)
    print(f"✅ 读取 {len(sequences)} 条序列")

    # 2.特征提取
    df_features = extract_features(sequences)
    print(f"✅ 特征矩阵: {df_features.shape[0]} 行 × {df_features.shape[1]} 列")

    # 3.相似性搜索
    df_hits = search_similar(df_features, email, api_key)

    # 显示搜索结果
    print("\nTop5搜索结果预览:")
    print(df_hits.head(10))

    # 4.保存
    df_features.to_csv("feature_matrix.csv", index=False, encoding="utf-8-sig")
    df_hits.to_csv("top5_results.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存")
    return df_features, df_hits


# 提问：函数后面加pass啥意思？
# pass 是 Python 的占位符，表示"这里什么都不做"。写框架时函数体还不能为空，先写个 pass 让代码不报错，等实现时再删掉。
# 所以 load_sequence 和 extract_features 里的 pass 应该删掉，改成 return

if __name__ == "__main__":  # ✅ 修正：之前是 "_main_"，应该是 "__main__"
    input_fasta = r"D:\Mary.yu\study\code\antibodies.fasta"
    email = 'yumengzhang77@gmail.com'
    api_key = 'fe85877f926cadde12d748b2cee17b570b08'
    run_pipeline(input_fasta, email, api_key)

# 提问：if __name__== "_main_"这个是什么格式？
# 这是 Python 的入口判断。当你直接运行 python Day_5.py 时，这行后面的代码会执行；
# 当被别人 import 时，这行后面的代码不会执行。

# 我的 pipeline 分四步。
# 第一步用 SeqIO 读 FASTA，返回（你实际跑出来的序列数）条序列。
# 第二步用 ProtParam 提取（你实际跑出来的特征列数）个特征，拼成 DataFrame。
# 第三步用 NCBI Entrez 做相似性搜索，每条序列取前 20 个氨基酸当搜索词，ESearch 加 ESummary 拿到 Top5 的物种、长度和标题。
# 第四步输出两个 CSV。整个流程从命令行一键跑通。没有输出