import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

# 导入项目模块
import config_umppi
import umppi
import data_loader

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class InteractionHeatmapGenerator:
    def __init__(self, model_path, config=None):
        """初始化热图生成器"""
        self.config = config or config_umppi.get_train_config()
        self.device = torch.device("cuda" if self.config.cuda else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        """加载预训练模型"""
        model = umppi.Model(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def prepare_pairs(self, peptides, targets):
        """生成多肽与靶点的全配对组合"""
        pairs = []
        for pep_name, pep_seq in peptides.items():
            for target_name, target_seq in targets.items():
                # 序列预处理（根据模型要求进行填充或截断）
                processed_pep = data_loader.get_same_len(pep_seq, self.config.pad_pep_len)
                processed_target = data_loader.get_same_len(target_seq, self.config.pad_prot_len)
                pairs.append({
                    'peptide_name': pep_name,
                    'peptide_seq': processed_pep,
                    'target_name': target_name,
                    'target_seq': processed_target
                })
        return pairs
    
    def extract_features(self, peptide_seq, target_seq):
        """提取多肽和靶点的特征"""
        # 这里需要根据实际数据预处理流程补充特征提取代码
        # 简化版示例，实际应使用data_loader中的特征提取逻辑
        pep_mask = data_loader.get_mask(peptide_seq, self.config.pad_pep_len)
        target_mask = data_loader.get_mask(target_seq, self.config.pad_prot_len)
        
        # 注意：实际应用中需要补充完整的特征提取
        return {
            'peptide': peptide_seq,
            'target': target_seq,
            'pep_mask': pep_mask,
            'target_mask': target_mask
        }
    
    def predict_interactions(self, peptides, targets):
        """预测所有多肽-靶点对的结合概率"""
        pairs = self.prepare_pairs(peptides, targets)
        results = []
        
        with torch.no_grad():
            for pair in tqdm(pairs, desc="预测结合概率"):
                # 提取特征
                features = self.extract_features(pair['peptide_seq'], pair['target_seq'])
                
                # 准备模型输入（需要根据实际模型输入格式调整）
                # 这里是简化版，实际应使用完整的特征输入
                pep_seqs = [pair['peptide_seq']]
                prot_seqs = [pair['target_seq']]
                
                # 生成空的特征张量（实际应用中需要替换为真实特征）
                dummy = torch.zeros(1, self.config.pad_pep_len).long().to(self.device)
                
                # 模型预测
                pred_binary, _, _ = self.model.predict(
                    dummy, dummy, dummy, dummy, dummy, dummy, 
                    dummy.float(), dummy.float(), prot_seqs, pep_seqs
                )
                
                # 计算结合概率
                prob = torch.softmax(pred_binary, dim=1)[0, 1].item()
                
                results.append({
                    'peptide': pair['peptide_name'],
                    'target': pair['target_name'],
                    'probability': prob,
                    'peptide_seq': pair['peptide_seq'],
                    'target_seq': pair['target_seq']
                })
        
        return pd.DataFrame(results)
    
    def get_attention_weights(self, peptide_seq, target_seq):
        """获取注意力权重以识别关键结合残基"""
        # 注意：实际应用中需要修改模型以返回注意力权重
        # 这里是模拟数据
        pep_len = min(len(peptide_seq), self.config.pad_pep_len)
        target_len = min(len(target_seq), self.config.pad_prot_len)
        
        # 生成模拟的注意力权重
        pep_attn = np.random.rand(pep_len) * 0.8 + 0.2  # 0.2-1.0之间的随机值
        target_attn = np.random.rand(target_len) * 0.8 + 0.2
        
        return {
            'peptide_attention': pep_attn,
            'target_attention': target_attn
        }
    
    def visualize_top_pairs(self, results, top_n=50):
        """可视化Top N候选对的热图"""
        # 按结合概率排序并选择Top N
        top_pairs = results.sort_values('probability', ascending=False).head(top_n)
        
        # 构建热图数据
        heatmap_data = top_pairs.pivot(
            index='peptide', 
            columns='target', 
            values='probability'
        ).fillna(0)
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', ['#f7fbff', '#abd0e6', '#3787c0', '#0d4b87']
        )
        
        # 绘制热图
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            heatmap_data, 
            annot=False, 
            cmap=cmap, 
            vmin=0, 
            vmax=1,
            cbar_kws={'label': '结合概率'}
        )
        plt.title(f'多肽-靶点结合概率热图 (Top {top_n})', fontsize=16)
        plt.tight_layout()
        plt.savefig('interaction_heatmap.png', dpi=300)
        plt.show()
        
        return top_pairs
    
    def visualize_key_residues(self, top_pairs, num_visualize=5):
        """可视化关键结合残基"""
        # 选择结合概率最高的几个配对进行可视化
        for i, (_, pair) in enumerate(top_pairs.head(num_visualize).iterrows()):
            attn = self.get_attention_weights(pair['peptide_seq'], pair['target_seq'])
            
            # 绘制多肽关键残基
            plt.figure(figsize=(12, 4))
            
            # 多肽注意力可视化
            plt.subplot(1, 2, 1)
            pep_seq = pair['peptide_seq'].replace('<pad>', '')
            pep_attn = attn['peptide_attention'][:len(pep_seq)]
            plt.bar(range(len(pep_seq)), pep_attn, color='skyblue')
            plt.xticks(range(len(pep_seq)), list(pep_seq), rotation=90)
            plt.title(f'{pair["peptide"]} 关键残基 (结合概率: {pair["probability"]:.4f})')
            plt.ylabel('注意力权重')
            
            # 靶点关键残基（只显示前50个残基）
            plt.subplot(1, 2, 2)
            target_seq = pair['target_seq'].replace('<pad>', '')[:50]  # 只显示前50个
            target_attn = attn['target_attention'][:len(target_seq)]
            plt.bar(range(len(target_seq)), target_attn, color='lightgreen')
            plt.xticks(range(len(target_seq)), list(target_seq), rotation=90)
            plt.title(f'{pair["target"]} 关键残基 (前50个)')
            plt.ylabel('注意力权重')
            
            plt.tight_layout()
            plt.savefig(f'key_residues_{i}.png', dpi=300)
            plt.show()

# 使用示例
if __name__ == "__main__":
    # 1. 配置模型路径和参数
    model_path = "./saved_models/esm_in_conv3/esm_0.0001_new_protein_fold0.pth"  # 替换为实际模型路径
    generator = InteractionHeatmapGenerator(model_path)
    
    # 2. 准备输入数据（示例数据，实际应替换为真实序列）
    # GLP-1类似物示例
    peptides = {
        "GLP-1": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
        "Semaglutide": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG",
        "Liraglutide": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
    }
    
    # 潜在靶点库示例
    targets = {
        "GCGR": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "GLP1R": "MSTEGSVEDKAAAATVQEQRAAAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAVQGPAGAGAGAGAGAGSGPPPAQASSSGAGAGAGAGAGAGAGAGAGAGAGAGAGAG",
        "GIPR": "MAVLLLALWLPLLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "SSTR2": "MAAGTLQPAFLLLLTLLGLALAVGQVRRRERESLEERNALQETQPPQSPPGTSWEAAEPGPGEAGEAGAGAGAGGGPGPGEAGEAGEAGEPGPGEAGEAGEAGEPGPG",
        "CD36": "MGPRWRLLLLAALLALWAPAPAHAEVIQSVNATWVFGVGFQILLTGLSVYLLAVGFYQHLRKLRPPEVWQHVSLAFGFVCLAILAGPGLVLWAVGFYSLDVLTFVGFV"
    }
    
    # 3. 预测所有配对的结合概率
    results = generator.predict_interactions(peptides, targets)
    
    # 4. 生成Top 50候选对热图
    top_pairs = generator.visualize_top_pairs(results, top_n=50)
    
    # 5. 可视化关键结合残基
    generator.visualize_key_residues(top_pairs, num_visualize=5)