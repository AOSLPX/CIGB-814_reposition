import argparse
import os
import csv
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd

import config_umppi
import umppi


def pad_seq(seq: str, pad_len: int) -> str:
    if len(seq) <= pad_len:
        return seq + (pad_len - len(seq)) * '<pad>'
    return seq[:pad_len]


def make_dummy_features(config) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    pad_pep_len = config.pad_pep_len
    pad_prot_len = config.pad_prot_len
    X_pep = torch.zeros(1, pad_pep_len, dtype=torch.long)
    X_p = torch.zeros(1, pad_prot_len, dtype=torch.long)
    X_SS_pep = torch.zeros(1, pad_pep_len, dtype=torch.long)
    X_SS_p = torch.zeros(1, pad_prot_len, dtype=torch.long)
    X_2_pep = torch.zeros(1, pad_pep_len, dtype=torch.long)
    X_2_p = torch.zeros(1, pad_prot_len, dtype=torch.long)
    X_dense_pep = torch.zeros(1, pad_pep_len, 3, dtype=torch.float) 
    X_dense_p = torch.zeros(1, pad_prot_len, 23, dtype=torch.float)
    return X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p


def load_targets_from_csv(path: str) -> Dict[str, str]:
    df = pd.read_csv(path, encoding='gbk')
    # 期望列名: name, seq
    if 'name' not in df.columns or 'seq' not in df.columns:
        raise ValueError('targets CSV需要包含列: name, seq')
    targets = {row['name']: row['seq'] for _, row in df.iterrows() if isinstance(row['seq'], str) and len(row['seq']) > 0}
    return targets





def load_peptide(peptide: str, peptide_file: str = None) -> Tuple[str, str]:
    if peptide:
        return 'CIGB-814', peptide
    if peptide_file:
        # 文本文件: 第一行为名称; 第二行为序列
        with open(peptide_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) == 1:
            return 'CIGB-814', lines[0]
        if len(lines) >= 2:
            return lines[0], lines[1]
    raise ValueError('必须提供 --peptide 或 --peptide-file')


def run_reposition(peptide_name: str, peptide_seq: str, targets: Dict[str, str], model_path: str, output_dir: str, top_n: int):
    os.makedirs(output_dir, exist_ok=True)

    config = config_umppi.get_train_config()
    # 检测CUDA是否可用，如果不可用则强制使用CPU
    if config.cuda and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU模式")
        config.cuda = False
    device = torch.device('cuda' if config.cuda and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = umppi.Model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results: List[Dict] = []

    with torch.no_grad():
        for tgt_name, tgt_seq in targets.items():
            pep = pad_seq(peptide_seq, config.pad_pep_len)
            prot = pad_seq(tgt_seq, config.pad_prot_len)

            X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p = make_dummy_features(config)
            # 送GPU/CPU
            X_pep = X_pep.to(device)
            X_p = X_p.to(device)
            X_SS_pep = X_SS_pep.to(device)
            X_SS_p = X_SS_p.to(device)
            X_2_pep = X_2_pep.to(device)
            X_2_p = X_2_p.to(device)
            X_dense_pep = X_dense_pep.to(device)
            X_dense_p = X_dense_p.to(device)

            # 预测
            pred_binary, pred_prot_site, pred_pep_site = model.predict(
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p,
                [prot], [pep]
            )

            prob = torch.softmax(pred_binary, dim=1)[0, 1].item()

            # 位点分数（对齐到真实长度）
            real_pep_len = min(len(peptide_seq), config.pad_pep_len)
            real_prot_len = min(len(tgt_seq), config.pad_prot_len)
            pep_site_prob = torch.softmax(pred_pep_site[0, :real_pep_len], dim=1)[:, 1].detach().cpu().numpy()
            prot_site_prob = torch.softmax(pred_prot_site[0, :real_prot_len], dim=1)[:, 1].detach().cpu().numpy()

            results.append({
                'peptide': peptide_name,
                'peptide_seq': peptide_seq,
                'target': tgt_name,
                'target_seq': tgt_seq,
                'probability': prob,
                'pep_site_prob': pep_site_prob,
                'prot_site_prob': prot_site_prob,
            })

    # 排序与导出
    results_sorted = sorted(results, key=lambda x: x['probability'], reverse=True)
    csv_path = os.path.join(output_dir, f'{peptide_name}_reposition_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['peptide', 'target', 'probability'])
        for r in results_sorted:
            writer.writerow([r['peptide'], r['target'], f"{r['probability']:.6f}"])

    # 位点明细（仅导出Top N）
    site_detail_path = os.path.join(output_dir, f'{peptide_name}_top{top_n}_sites.csv')
    with open(site_detail_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'target', 'type', 'position', 'score'])
        for rank, r in enumerate(results_sorted[:top_n], start=1):
            for i, s in enumerate(r['pep_site_prob'], start=1):
                writer.writerow([rank, r['target'], 'peptide', i, f"{float(s):.6f}"])
            for i, s in enumerate(r['prot_site_prob'], start=1):
                writer.writerow([rank, r['target'], 'protein', i, f"{float(s):.6f}"])

    print(f'已保存: {csv_path}')
    print(f'已保存: {site_detail_path}')


def main():
    parser = argparse.ArgumentParser(description='CIGB-814 药物重定位与靶点位点预测')
    parser.add_argument('--peptide', type=str, default='', help='CIGB-814 序列，或留空配合 --peptide-file 使用')
    parser.add_argument('--peptide-file', type=str, default='', help='包含多肽名称与序列的txt文件（1行名称，2行序列）')
    parser.add_argument('--targets-csv', type=str, required=True, help='候选靶点CSV，列: name, seq')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径，如 saved_models/esm_in_conv3/esm_0.0001_new_protein_fold0.pth')
    parser.add_argument('--output-dir', type=str, default='cigb814_outputs', help='输出目录')
    parser.add_argument('--topN', type=int, default=20, help='导出位点明细的Top N配对')

    args = parser.parse_args()

    pep_name, pep_seq = load_peptide(args.peptide, args.peptide_file)
    targets = load_targets_from_csv(args.targets_csv)
    run_reposition(pep_name, pep_seq, targets, args.model_path, args.output_dir, args.topN)


if __name__ == '__main__':
    main()



