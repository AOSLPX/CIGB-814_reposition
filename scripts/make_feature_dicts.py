# make_feature_dicts.py
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_loader import DataLoader
from sklearn.decomposition import PCA
import os

# ------------------- 参数 -------------------
CSV_PATH = "datasets/Dataset_all_balanced_new_2.csv"
OUT_DIR  = "data_feature_dict"
MAX_LEN_PROTEIN = 1024
MAX_LEN_PEPTIDE = 50
PCA_COMPONENTS = 128

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- 读取数据 -------------------
df = pd.read_csv(CSV_PATH)
proteins = df['prot_seq'].dropna().unique().tolist()
peptides = df['pep_seq'].dropna().unique().tolist()

loader = DataLoader(max_len_protein=MAX_LEN_PROTEIN,
                    max_len_peptide=MAX_LEN_PEPTIDE,
                    ss_pred=True)  # 自动用 ESM 或 PSIPRED

# ------------------- 1. 原始特征 -------------------
print("生成 protein_feature_dict / peptide_feature_dict")
protein_feat_dict = {}
peptide_feat_dict = {}
for seq in tqdm(proteins, desc="Protein"):
    protein_feat_dict[seq] = loader.get_feature_dict(seq, is_protein=True)
for seq in tqdm(peptides, desc="Peptide"):
    peptide_feat_dict[seq] = loader.get_feature_dict(seq, is_protein=False)

# ------------------- 2. 二级结构特征 -------------------
print("生成 SS 特征字典")
protein_ss_dict = {seq: loader.ss_to_vector(loader.predict_ss(seq, is_protein=True), MAX_LEN_PROTEIN)
                   for seq in tqdm(proteins, desc="Protein SS")}
peptide_ss_dict = {seq: loader.ss_to_vector(loader.predict_ss(seq, is_protein=False), MAX_LEN_PEPTIDE)
                   for seq in tqdm(peptides, desc="Peptide SS")}

# ------------------- 3. 2-mer 频率 -------------------
def kmer_freq(seq, k=2):
    if len(seq) < k: return np.zeros(400)
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    freq = np.zeros(400)
    aa_to_idx = {aa:i for i,aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    for kmer in kmers:
        idx = aa_to_idx[kmer[0]] * 20 + aa_to_idx[kmer[1]]
        freq[idx] += 1
    return freq / len(kmers)

protein_2_dict = {seq: kmer_freq(seq) for seq in tqdm(proteins, desc="Protein 2-mer")}
peptide_2_dict = {seq: kmer_freq(seq) for seq in tqdm(peptides, desc="Peptide 2-mer")}

# ------------------- 4. Dense 特征（PCA） -------------------
def concat_features(feat, ss, k2):
    return np.concatenate([feat.reshape(-1), ss.reshape(-1), k2], axis=0)

X_pro = np.array([concat_features(protein_feat_dict[s], protein_ss_dict[s], protein_2_dict[s]) for s in proteins])
X_pep = np.array([concat_features(peptide_feat_dict[s], peptide_ss_dict[s], peptide_2_dict[s]) for s in peptides])

pca_pro = PCA(n_components=PCA_COMPONENTS).fit(X_pro)
pca_pep = PCA(n_components=PCA_COMPONENTS).fit(X_pep)

protein_dense_dict = {seq: pca_pro.transform([concat_features(protein_feat_dict[seq], protein_ss_dict[seq], protein_2_dict[seq])])[0]
                      for seq in proteins}
peptide_dense_dict = {seq: pca_pep.transform([concat_features(peptide_feat_dict[seq], peptide_ss_dict[seq], peptide_2_dict[seq])])[0]
                      for seq in peptides}

# ------------------- 保存 -------------------
def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

save(protein_feat_dict, f"{OUT_DIR}/protein_feature_dict")
save(peptide_feat_dict, f"{OUT_DIR}/peptide_feature_dict")
save(protein_ss_dict, f"{OUT_DIR}/protein_ss_feature_dict")
save(peptide_ss_dict, f"{OUT_DIR}/peptide_ss_feature_dict")
save(protein_2_dict, f"{OUT_DIR}/protein_2_feature_dict")
save(peptide_2_dict, f"{OUT_DIR}/peptide_2_feature_dict")
save(protein_dense_dict, f"{OUT_DIR}/protein_dense_feature_dict")
save(peptide_dense_dict, f"{OUT_DIR}/peptide_dense_feature_dict")
save(pca_pro, f"{OUT_DIR}/pca_protein.pkl")
save(pca_pep, f"{OUT_DIR}/pca_peptide.pkl")

print("所有特征字典生成完成！")