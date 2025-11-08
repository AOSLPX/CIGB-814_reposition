import os
import argparse
import random
from typing import List, Tuple

import torch
import torch.utils.data as Data
import numpy as np

import config_umppi
import umppi
import torch.nn.functional as F


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA)}  # 0 作为 pad/未知


def pad_seq(seq: str, pad_len: int) -> str:
    if len(seq) <= pad_len:
        return seq + (pad_len - len(seq)) * '<pad>'
    return seq[:pad_len]


def aa_to_ids(seq: str, pad_len: int) -> np.ndarray:
    ids = [AA_TO_IDX.get(c, 0) for c in seq if c != '<' and c != 'p' and c != 'a' and c != 'd' and c != '>']
    ids = ids[:pad_len]
    if len(ids) < pad_len:
        ids += [0] * (pad_len - len(ids))
    return np.array(ids, dtype=np.int64)


def build_bs_vec(bs_str: str, N: int, positive_flag: bool) -> Tuple[np.ndarray, float]:
    # 返回 (bs_vector, flag)
    if not positive_flag:
        return np.zeros(N, dtype=np.int64), 0.0
    # 正样本
    if bs_str == '-99999':
        return np.zeros(N, dtype=np.int64), 0.0
    if bs_str == 'NoBinding':
        return np.zeros(N, dtype=np.int64), 0.0
    idxs = [int(x) for x in bs_str.split(',') if x.isdigit()]
    idxs = [x for x in idxs if x < N]
    vec = np.zeros(N, dtype=np.int64)
    for i in idxs:
        vec[i] = 1
    return vec, 1.0


class QuickDataset(Data.Dataset):
    def __init__(self, lines: List[str], pad_pep_len: int, pad_prot_len: int):
        self.pad_pep_len = pad_pep_len
        self.pad_prot_len = pad_prot_len

        X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p = [], [], [], [], [], []
        X_dense_pep, X_dense_p = [], []
        pep_sequence, prot_sequence = [], []
        X_pep_mask, X_bs_flag, X_bs = [], [], []
        X_prot_mask, X_prot_bs_flag, X_prot_bs = [], [], []
        labels = []

        for line in lines:
            seq, peptide, peptide_ss, seq_ss, label, pep_bs, prot_bs = line.strip().split('\t')
            label_i = int(label)

            pep_seq_padded = pad_seq(peptide, pad_pep_len)
            prot_seq_padded = pad_seq(seq, pad_prot_len)

            pep_ids = aa_to_ids(pep_seq_padded, pad_pep_len)
            prot_ids = aa_to_ids(prot_seq_padded, pad_prot_len)

            # 简化：SS/2-mer 用 0；dense 用 0（但需匹配输入维度）
            X_pep.append(pep_ids)
            X_p.append(prot_ids)
            X_SS_pep.append(np.zeros(pad_pep_len, dtype=np.int64))
            X_SS_p.append(np.zeros(pad_prot_len, dtype=np.int64))
            X_2_pep.append(np.zeros(pad_pep_len, dtype=np.int64))
            X_2_p.append(np.zeros(pad_prot_len, dtype=np.int64))

            X_dense_pep.append(np.zeros((pad_pep_len, 3), dtype=np.float32))
            X_dense_p.append(np.zeros((pad_prot_len, 23), dtype=np.float32))

            pep_sequence.append(pep_seq_padded)
            prot_sequence.append(prot_seq_padded)

            # masks
            pep_mask = np.zeros(pad_pep_len, dtype=np.int64)
            pep_mask[:min(len(peptide), pad_pep_len)] = 1
            prot_mask = np.zeros(pad_prot_len, dtype=np.int64)
            prot_mask[:min(len(seq), pad_prot_len)] = 1
            X_pep_mask.append(pep_mask)
            X_prot_mask.append(prot_mask)

            # binding sites & flags
            pep_vec, pep_flag = build_bs_vec(pep_bs, pad_pep_len, label_i == 1)
            prot_vec, prot_flag = build_bs_vec(prot_bs, pad_prot_len, label_i == 1)
            X_bs.append(pep_vec)
            X_bs_flag.append(pep_flag)
            X_prot_bs.append(prot_vec)
            X_prot_bs_flag.append(prot_flag)

            labels.append(label_i)

        # to tensors
        self.X_pep = torch.LongTensor(np.stack(X_pep))
        self.X_p = torch.LongTensor(np.stack(X_p))
        self.X_SS_pep = torch.LongTensor(np.stack(X_SS_pep))
        self.X_SS_p = torch.LongTensor(np.stack(X_SS_p))
        self.X_2_pep = torch.LongTensor(np.stack(X_2_pep))
        self.X_2_p = torch.LongTensor(np.stack(X_2_p))
        self.X_dense_pep = torch.tensor(np.stack(X_dense_pep), dtype=torch.float32)
        self.X_dense_p = torch.tensor(np.stack(X_dense_p), dtype=torch.float32)
        self.pep_sequence = np.array(pep_sequence)
        self.prot_sequence = np.array(prot_sequence)
        self.X_pep_mask = torch.LongTensor(np.stack(X_pep_mask))
        self.X_bs_flag = torch.LongTensor(np.array(X_bs_flag))
        self.X_bs = torch.LongTensor(np.stack(X_bs))
        self.labels = torch.LongTensor(np.array(labels))
        self.X_prot_mask = torch.LongTensor(np.stack(X_prot_mask))
        self.X_prot_bs_flag = torch.LongTensor(np.array(X_prot_bs_flag))
        self.X_prot_bs = torch.LongTensor(np.stack(X_prot_bs))

    def __len__(self):
        return len(self.X_pep)

    def __getitem__(self, idx):
        return self.X_pep[idx], self.X_p[idx], self.X_SS_pep[idx], self.X_SS_p[idx], self.X_2_pep[idx], self.X_2_p[idx], \
            self.X_dense_pep[idx], self.X_dense_p[idx], self.pep_sequence[idx], self.prot_sequence[idx], \
            self.X_pep_mask[idx], self.X_bs_flag[idx], self.X_bs[idx], self.labels[idx], \
            self.X_prot_mask[idx], self.X_prot_bs_flag[idx], self.X_prot_bs[idx]


def split_lines(lines: List[str], train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.Random(seed).shuffle(lines)
    n = len(lines)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = lines[:n_train]
    valid = lines[n_train:n_train + n_valid]
    test = lines[n_train + n_valid:]
    return train, valid, test


def evaluate(loader, model, criterion, device, config):
    model.eval()
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    avg_loss = 0
    with torch.no_grad():
        for batch in loader:
            X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_seqs, prot_seqs, \
                X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs = batch
            X_pep = X_pep.to(device)
            X_p = X_p.to(device)
            X_SS_pep = X_SS_pep.to(device)
            X_SS_p = X_SS_p.to(device)
            X_2_pep = X_2_pep.to(device)
            X_2_p = X_2_p.to(device)
            X_dense_pep = X_dense_pep.to(device)
            X_dense_p = X_dense_p.to(device)
            labels = labels.view(-1).to(device)

            pred_binary, _, _ = model.predict(
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p,
                list(prot_seqs), list(pep_seqs)
            )

            pred_prob_all = F.softmax(pred_binary, dim=1)
            loss = criterion(pred_binary, labels)
            avg_loss += loss
            p_class = torch.max(pred_prob_all, 1)[1]
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, labels.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_all[:, 1].view(-1)])

    avg_loss /= len(loader)
    # 简化：仅返回平均损失与 ACC
    acc = (label_pred == label_real).float().mean().item()
    return avg_loss.item(), acc


def main():
    parser = argparse.ArgumentParser(description='Quick Train from Dataset_all_balanced_new_2')
    parser.add_argument('--data-path', type=str, default=r'F:\学习工作\结题\SummerWORK2025\Dataset_all_balanced_new_2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-dir', type=str, default='./saved_models/esm_in_conv3')
    args = parser.parse_args()

    config = config_umppi.get_train_config()
    config.batch_size = args.batch_size
    # 检测CUDA是否可用，如果不可用则强制使用CPU
    if config.cuda and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU模式")
        config.cuda = False
    device = torch.device('cuda' if config.cuda and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("=" * 60)
    print("步骤 1/5: 读取数据文件...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过表头
    print(f"  读取到 {len(lines)} 条数据")

    print("步骤 2/5: 划分训练/验证/测试集...")
    train_lines, valid_lines, test_lines = split_lines(lines)
    print(f"  训练集: {len(train_lines)} 条")
    print(f"  验证集: {len(valid_lines)} 条")
    print(f"  测试集: {len(test_lines)} 条")

    print("步骤 3/5: 构建数据集（这可能需要几分钟）...")
    train_ds = QuickDataset(train_lines, config.pad_pep_len, config.pad_prot_len)
    print("  训练集构建完成")
    valid_ds = QuickDataset(valid_lines, config.pad_pep_len, config.pad_prot_len)
    print("  验证集构建完成")
    test_ds = QuickDataset(test_lines, config.pad_pep_len, config.pad_prot_len)
    print("  测试集构建完成")

    print("步骤 4/5: 创建数据加载器...")
    train_loader = Data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    valid_loader = Data.DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = Data.DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False)
    print(f"  训练批次数: {len(train_loader)}")

    print("步骤 5/5: 初始化模型（首次运行会下载ESM模型，请耐心等待）...")
    model = umppi.Model(config).to(device)
    print("  模型初始化完成！")
    print("=" * 60)
    criterion = torch.nn.CrossEntropyLoss()

    bert_params, other_params = [], []
    for name, para in model.named_parameters():
        if para.requires_grad:
            if "BERT" in name:
                bert_params.append(para)
            else:
                other_params.append(para)
    if not bert_params and not other_params:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": 1e-5},
            {"params": other_params, "lr": 1e-3},
        ])

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"esm_{config.clu_thre}_{config.setting}_fold0.pth")

    print("\n开始训练...")
    print("=" * 60)
    for ep in range(1, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")
        model.train()
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_seqs, prot_seqs, \
                X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs = batch

            X_pep = X_pep.to(device)
            X_p = X_p.to(device)
            X_SS_pep = X_SS_pep.to(device)
            X_SS_p = X_SS_p.to(device)
            X_2_pep = X_2_pep.to(device)
            X_2_p = X_2_p.to(device)
            X_dense_pep = X_dense_pep.to(device)
            X_dense_p = X_dense_p.to(device)
            labels = labels.view(-1).to(device)

            pred_binary, _, _ = model.predict(
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p,
                list(prot_seqs), list(pep_seqs)
            )

            loss = criterion(pred_binary, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 每10个batch输出一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch [{batch_idx + 1}/{total_batches}] - Loss: {loss.item():.4f}")

        print(f"  验证中...")
        val_loss, val_acc = evaluate(valid_loader, model, criterion, device, config)
        print(f"  Epoch {ep} 结果: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"保存新最佳权重: {save_path}")

    print("\n" + "=" * 60)
    print("训练完成！开始测试...")
    test_loss, test_acc = evaluate(test_loader, model, criterion, device, config)
    print(f"测试结果: loss={test_loss:.4f} acc={test_acc:.4f}")
    print("=" * 60)
    print(f"\n✅ 模型权重已保存至: {save_path}")
    print(f"   最佳验证准确率: {best_acc:.4f}")
    print(f"   测试准确率: {test_acc:.4f}")
    print("\n现在可以使用此权重运行 CIGB-814 重定位脚本！")


if __name__ == '__main__':
    main()



