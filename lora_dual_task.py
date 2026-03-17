#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 LoRA 的中文微博双任务分类模型
任务1: 情感极性 (负面/中性/正面)
任务2: 话题立场 (反对/中立/支持)
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import logging
import argparse
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 配置类 ====================

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_length: int = 128
    hidden_size: int = 768
    
    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "value"]  # 在 Attention 的 Q/V 矩阵注入 LoRA


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # 任务权重
    sentiment_weight: float = 0.5
    stance_weight: float = 0.5
    
    # 保存路径
    output_dir: str = "./output"
    save_steps: int = 500


# ==================== 数据预处理 ====================

class DataProcessor:
    """数据处理器：加载和预处理微博数据"""
    
    # 标签映射
    SENTIMENT_LABELS = {'negative': 0, 'neutral': 1, 'positive': 2}
    STANCE_LABELS = {'against': 0, 'neutral': 1, 'favor': 2}
    
    SENTIMENT_ID2LABEL = {v: k for k, v in SENTIMENT_LABELS.items()}
    STANCE_ID2LABEL = {v: k for k, v in STANCE_LABELS.items()}
    
    # 立场推断关键词（扩展版）
    SUPPORT_KEYWORDS = ['支持', '赞同', '同意', '赞', '看好', '拥护', '坚定', '必须', 
                       '肯定', '正确', '好', '棒', '优秀', '喜欢', '爱', '推荐', 
                       '值得', '给力', '完美', '满意', '真香', '好评', '点赞']
    AGAINST_KEYWORDS = ['反对', '抗议', '拒绝', '否定', '错误', '差', '坏', '垃圾', 
                       '讨厌', '恶心', '失望', '愤怒', '不满', '抵制', '不建议', 
                       '千万别', '坑', '骗', '假', '烂', '差劲', '后悔', '差评']
    NEUTRAL_KEYWORDS = ['一般', '还行', '凑合', '普通', '一般般', '看看', '观望', 
                       '待定', '考虑', '再说', '可能', '也许', '不知道', '不清楚',
                       '中立', '客观', '理性', '辩证']
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清洗微博文本"""
        import re
        
        if not isinstance(text, str):
            text = str(text)
            
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 去除@用户
        text = re.sub(r'@[\w\-]+', '', text)
        # 去除话题标签但保留内容 #话题#
        text = re.sub(r'#([^#]+)#', r'\1', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除特殊符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）]', ' ', text)
        
        return text.strip()
    
    @classmethod
    def infer_stance_from_text(cls, text: str, sentiment_label: int, index: int = 0) -> int:
        """
        改进的立场推断 - 基于关键词和情感，增加随机性避免全中立
        返回: 0=反对, 1=中立, 2=支持
        """
        text = str(text).lower()
        
        # 统计关键词（允许重复计数，考虑词频）
        support_count = sum(text.count(kw) for kw in cls.SUPPORT_KEYWORDS)
        against_count = sum(text.count(kw) for kw in cls.AGAINST_KEYWORDS)
        neutral_count = sum(text.count(kw) for kw in cls.NEUTRAL_KEYWORDS)
        
        # 设置随机种子确保可复现，但基于index有变化
        rng = random.Random(index + hash(text) % 10000)
        
        # 基于明确关键词判断
        if support_count > against_count and support_count > 0:
            return 2  # 明确支持
        elif against_count > support_count and against_count > 0:
            return 0  # 明确反对
        elif neutral_count > 0:
            return 1  # 明确中立
        
        # 无明确关键词时，基于情感随机分配，避免全中立
        if sentiment_label == 2:  # 正面情感
            # 70%支持, 20%中立, 10%反对（讽刺）
            r = rng.random()
            if r < 0.7:
                return 2
            elif r < 0.9:
                return 1
            else:
                return 0
        elif sentiment_label == 0:  # 负面情感
            # 60%反对, 30%中立, 10%支持（批评但支持改进）
            r = rng.random()
            if r < 0.6:
                return 0
            elif r < 0.9:
                return 1
            else:
                return 2
        else:  # 中性情感
            # 40%中立, 30%支持, 30%反对
            r = rng.random()
            if r < 0.4:
                return 1
            elif r < 0.7:
                return 2
            else:
                return 0
    
    @classmethod
    def load_from_json(cls, file_path: str) -> pd.DataFrame:
        """
        从JSON加载数据
        期望格式: [{"text": "...", "sentiment": "positive", "stance": "favor"}, ...]
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['text'] = df['text'].apply(cls.clean_text)
        
        # 转换标签为数字
        df['sentiment_label'] = df['sentiment'].map(cls.SENTIMENT_LABELS)
        df['stance_label'] = df['stance'].map(cls.STANCE_LABELS)
        
        # 过滤无效数据
        df = df.dropna(subset=['sentiment_label', 'stance_label'])
        
        return df
    
    @classmethod
    def load_from_csv(cls, file_path: str, text_col: str = 'text',
                      sentiment_col: str = 'sentiment', stance_col: str = 'stance',
                      auto_infer_stance: bool = True) -> pd.DataFrame:
        """
        从CSV加载数据，自动检测编码
        如果缺少立场标签，可以基于规则自动推断
        """
        # 尝试多种编码读取
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'ansi', 'latin1']
        
        df = None
        used_encoding = None
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                used_encoding = enc
                logger.info(f"使用 {enc} 编码成功读取CSV")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"无法识别文件编码，请确保文件为UTF-8或GBK格式: {file_path}")
        
        # 标准化列名（处理常见的列名变体）
        column_mapping = {
            'review': 'text',
            'content': 'text',
            'sentence': 'text',
            'comment': 'text',
            '微博内容': 'text',
            'emotion': 'sentiment',
            '情感': 'sentiment',
            'label': 'sentiment',
            '情感标签': 'sentiment',
            'attitude': 'stance',
            '态度': 'stance',
            '立场': 'stance',
            'stance_label': 'stance'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保有text列
        if 'text' not in df.columns:
            # 尝试使用第一列作为text
            first_col = df.columns[0]
            logger.warning(f"未找到'text'列，使用第一列'{first_col}'作为文本")
            df = df.rename(columns={first_col: 'text'})
        
        # 清洗文本
        df['text'] = df['text'].apply(cls.clean_text)
        
        # 处理情感标签
        if 'sentiment' not in df.columns and 'sentiment_label' not in df.columns:
            raise ValueError("CSV中缺少情感标签列（sentiment或sentiment_label）")
        
        if 'sentiment' in df.columns:
            if df['sentiment'].dtype == 'object':
                df['sentiment_label'] = df['sentiment'].map(cls.SENTIMENT_LABELS)
            else:
                df['sentiment_label'] = df['sentiment']
        
        # 处理立场标签
        has_stance = 'stance' in df.columns or 'stance_label' in df.columns
        
        if has_stance:
            if 'stance' in df.columns:
                if df['stance'].dtype == 'object':
                    df['stance_label'] = df['stance'].map(cls.STANCE_LABELS)
                else:
                    df['stance_label'] = df['stance']
        elif auto_infer_stance:
            # 自动推断立场标签（使用改进的规则）
            logger.info("CSV中缺少立场标签，基于规则自动推断...")
            df['stance_label'] = df.apply(
                lambda row: cls.infer_stance_from_text(
                    row['text'], 
                    row['sentiment_label'],
                    index=row.name  # 传入index确保随机性
                ),
                axis=1
            )
            logger.info(f"立场推断完成，分布:\n{df['stance_label'].value_counts().sort_index()}")
        else:
            # 全部设为中立
            logger.warning("CSV中缺少立场标签，全部设为中立(1)")
            df['stance_label'] = 1
        
        # 过滤无效数据
        df = df.dropna(subset=['sentiment_label', 'stance_label'])
        
        # 确保标签为整数
        df['sentiment_label'] = df['sentiment_label'].astype(int)
        df['stance_label'] = df['stance_label'].astype(int)
        
        return df
    
    @classmethod
    def create_sample_data(cls, num_samples: int = 1000) -> pd.DataFrame:
        """创建示例数据（用于测试）"""
        np.random.seed(42)
        random.seed(42)
        
        sample_texts = [
            "这个产品真的太棒了，完全超出预期！",
            "一般般吧，没什么特别的感受。",
            "太差了，完全不值得购买，浪费钱。",
            "支持这个政策，对大家都好。",
            "反对这种做法，会伤害很多人利益。",
            "持中立态度，观望后续发展。",
            "很喜欢这个功能，设计得很人性化。",
            "没什么感觉，用用看吧。",
            "非常失望，和宣传的差距太大。",
            "坚决支持，这是正确的决定！"
        ]
        
        data = []
        for i in range(num_samples):
            base_text = np.random.choice(sample_texts)
            # 随机扰动生成变体
            text = base_text + f" [样本{i}]"
            
            sentiment = np.random.choice(['negative', 'neutral', 'positive'])
            stance = np.random.choice(['against', 'neutral', 'favor'])
            
            data.append({
                'text': text,
                'sentiment': sentiment,
                'stance': stance
            })
        
        df = pd.DataFrame(data)
        df['sentiment_label'] = df['sentiment'].map(cls.SENTIMENT_LABELS)
        df['stance_label'] = df['stance'].map(cls.STANCE_LABELS)
        
        return df


# ==================== 数据集类 ====================

class WeiboDualDataset(Dataset):
    """微博双任务数据集"""
    
    def __init__(self, texts: List[str], sentiment_labels: List[int], 
                 stance_labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.stance_labels = stance_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            'stance_label': torch.tensor(self.stance_labels[idx], dtype=torch.long),
            'text': text  # 保留原文用于调试
        }


# ==================== 模型定义 ====================

class LoRADualTaskModel(nn.Module):
    """
    LoRA双任务分类模型
    - 冻结预训练BERT权重
    - 只训练LoRA参数和任务头
    """
    
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config
        
        # 加载预训练配置
        self.bert_config = AutoConfig.from_pretrained(model_config.model_name)
        
        # 加载基础模型
        self.base_model = AutoModel.from_pretrained(model_config.model_name)
        
        # 冻结所有预训练参数（关键：不训练原始权重）
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.target_modules,
            bias="none",
        )
        
        # 应用LoRA：只添加可训练的LoRA矩阵
        self.lora_model = get_peft_model(self.base_model, lora_config)
        
        # 计算可训练参数数量（兼容新旧版本PEFT）
        def count_parameters(model):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            return trainable, total
        
        trainable_params, all_params = count_parameters(self.lora_model)
        logger.info(f"LoRA可训练参数: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.4f}%)")
        
        # 双任务分类头（独立参数，都需要训练）
        self.dropout = nn.Dropout(0.1)
        
        # 情感分类头
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(model_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # 负面/中性/正面
        )
        
        # 立场分类头
        self.stance_classifier = nn.Sequential(
            nn.Linear(model_config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # 反对/中立/支持
        )
        
        # 初始化任务头权重
        self._init_weights(self.sentiment_classifier)
        self._init_weights(self.stance_classifier)
    
    def _init_weights(self, module):
        """初始化分类头权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_features: 是否返回特征向量（用于分析）
        Returns:
            dict: 包含logits和可选特征
        """
        # 通过LoRA-BERT获取表征
        outputs = self.lora_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 取[CLS] token的表征 [batch_size, hidden_size]
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # 双任务预测
        sentiment_logits = self.sentiment_classifier(pooled_output)
        stance_logits = self.stance_classifier(pooled_output)
        
        result = {
            'sentiment_logits': sentiment_logits,
            'stance_logits': stance_logits
        }
        
        if return_features:
            result['features'] = pooled_output
            result['hidden_states'] = outputs.hidden_states
            
        return result
    
    def save_pretrained(self, save_path: str):
        """保存模型（只保存LoRA权重和分类头）"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存LoRA权重
        self.lora_model.save_pretrained(os.path.join(save_path, "lora_adapter"))
        
        # 保存分类头
        torch.save({
            'sentiment_classifier': self.sentiment_classifier.state_dict(),
            'stance_classifier': self.stance_classifier.state_dict(),
        }, os.path.join(save_path, "task_heads.pt"))
        
        # 保存配置
        torch.save(self.config, os.path.join(save_path, "model_config.pt"))
        
        logger.info(f"模型已保存至: {save_path}")
    
    def load_pretrained(self, load_path: str):
        """加载模型"""
        # 加载LoRA权重
        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            os.path.join(load_path, "lora_adapter")
        )
        
        # 加载分类头
        heads_path = os.path.join(load_path, "task_heads.pt")
        checkpoint = torch.load(heads_path, map_location='cpu')
        self.sentiment_classifier.load_state_dict(checkpoint['sentiment_classifier'])
        self.stance_classifier.load_state_dict(checkpoint['stance_classifier'])
        
        logger.info(f"模型已从 {load_path} 加载")


# ==================== 训练器 ====================

class DualTaskTrainer:
    """双任务训练器"""
    
    def __init__(self, model: LoRADualTaskModel, train_config: TrainingConfig, 
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.config = train_config
        self.device = device
        
        # 分离参数：LoRA参数和任务头参数
        lora_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif 'classifier' in name.lower():
                    head_params.append(param)
        
        # 使用不同的学习率
        self.optimizer = AdamW([
            {'params': lora_params, 'lr': train_config.learning_rate * 0.5},  # LoRA使用较小学习率
            {'params': head_params, 'lr': train_config.learning_rate}          # 任务头使用正常学习率
        ], weight_decay=train_config.weight_decay)
        
        self.global_step = 0
        self.best_score = 0.0
        
    def compute_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """计算双任务损失"""
        sentiment_logits = outputs['sentiment_logits']
        stance_logits = outputs['stance_logits']
        
        sentiment_labels = batch['sentiment_label'].to(self.device)
        stance_labels = batch['stance_label'].to(self.device)
        
        # 交叉熵损失
        loss_sent = F.cross_entropy(sentiment_logits, sentiment_labels)
        loss_stance = F.cross_entropy(stance_logits, stance_labels)
        
        # 加权求和
        total_loss = (self.config.sentiment_weight * loss_sent + 
                     self.config.stance_weight * loss_stance)
        
        return total_loss, {
            'sentiment_loss': loss_sent.item(),
            'stance_loss': loss_stance.item()
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        sentiment_losses = []
        stance_losses = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(outputs, batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            sentiment_losses.append(loss_dict['sentiment_loss'])
            stance_losses.append(loss_dict['stance_loss'])
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sent': f"{loss_dict['sentiment_loss']:.4f}",
                'stance': f"{loss_dict['stance_loss']:.4f}"
            })
        
        return {
            'avg_loss': total_loss / len(dataloader),
            'avg_sentiment_loss': np.mean(sentiment_losses),
            'avg_stance_loss': np.mean(stance_losses)
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        all_sentiment_preds = []
        all_sentiment_labels = []
        all_stance_preds = []
        all_stance_labels = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            # 预测
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
            stance_preds = torch.argmax(outputs['stance_logits'], dim=1).cpu().numpy()
            
            all_sentiment_preds.extend(sentiment_preds)
            all_sentiment_labels.extend(batch['sentiment_label'].numpy())
            all_stance_preds.extend(stance_preds)
            all_stance_labels.extend(batch['stance_label'].numpy())
        
        # 计算指标
        metrics = {
            'sentiment_acc': accuracy_score(all_sentiment_labels, all_sentiment_preds),
            'sentiment_f1': f1_score(all_sentiment_labels, all_sentiment_preds, average='macro'),
            'stance_acc': accuracy_score(all_stance_labels, all_stance_preds),
            'stance_f1': f1_score(all_stance_labels, all_stance_preds, average='macro'),
        }
        
        # 计算平均分数（用于选择最佳模型）
        metrics['avg_score'] = (metrics['sentiment_f1'] + metrics['stance_f1']) / 2
        
        return metrics, (all_sentiment_labels, all_sentiment_preds, all_stance_labels, all_stance_preds)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: Optional[DataLoader] = None):
        """完整训练流程"""
        logger.info("开始训练...")
        
        for epoch in range(self.config.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
                       f"Sentiment: {train_metrics['avg_sentiment_loss']:.4f}, "
                       f"Stance: {train_metrics['avg_stance_loss']:.4f}")
            
            # 验证
            val_metrics, _ = self.evaluate(val_loader)
            logger.info(f"Val - Sentiment Acc/F1: {val_metrics['sentiment_acc']:.4f}/{val_metrics['sentiment_f1']:.4f}, "
                       f"Stance Acc/F1: {val_metrics['stance_acc']:.4f}/{val_metrics['stance_f1']:.4f}, "
                       f"Avg Score: {val_metrics['avg_score']:.4f}")
            
            # 保存最佳模型
            if val_metrics['avg_score'] > self.best_score:
                self.best_score = val_metrics['avg_score']
                save_path = os.path.join(self.config.output_dir, "best_model")
                self.model.save_pretrained(save_path)
                logger.info(f"保存最佳模型，Avg F1: {self.best_score:.4f}")
        
        # 最终测试
        if test_loader is not None:
            logger.info("\n最终测试...")
            # 加载最佳模型
            best_model_path = os.path.join(self.config.output_dir, "best_model")
            self.model.load_pretrained(best_model_path)
            
            test_metrics, test_results = self.evaluate(test_loader)
            logger.info(f"Test - Sentiment Acc/F1: {test_metrics['sentiment_acc']:.4f}/{test_metrics['sentiment_f1']:.4f}, "
                       f"Stance Acc/F1: {test_metrics['stance_acc']:.4f}/{test_metrics['stance_f1']:.4f}")
            
            # 详细报告
            self._print_detailed_report(*test_results)
    
    def _print_detailed_report(self, sent_labels, sent_preds, stance_labels, stance_preds):
        """打印详细分类报告 - 动态适应实际类别数"""
        logger.info("\n" + "="*50)
        
        # 情感报告 - 获取实际出现的类别
        sent_labels_set = sorted(set(sent_labels) | set(sent_preds))
        sent_target_names = ['负面', '中性', '正面']
        sent_names = [sent_target_names[i] for i in sent_labels_set if i < len(sent_target_names)]
        
        logger.info("情感分类报告:")
        if len(sent_labels_set) > 0:
            logger.info(classification_report(
                sent_labels, sent_preds,
                labels=sent_labels_set,
                target_names=sent_names,
                digits=4,
                zero_division=0  # 避免除零警告
            ))
        
        # 立场报告 - 获取实际出现的类别
        stance_labels_set = sorted(set(stance_labels) | set(stance_preds))
        stance_target_names = ['反对', '中立', '支持']
        stance_names = [stance_target_names[i] for i in stance_labels_set if i < len(stance_target_names)]
        
        logger.info("立场分类报告:")
        if len(stance_labels_set) > 0:
            logger.info(classification_report(
                stance_labels, stance_preds,
                labels=stance_labels_set,
                target_names=stance_names,
                digits=4,
                zero_division=0  # 避免除零警告
            ))
        logger.info("="*50)


# ==================== 推理类 ====================

class DualTaskPredictor:
    """双任务预测器"""
    
    def __init__(self, model_path: str, model_config: Optional[ModelConfig] = None, 
                 device: str = 'cuda'):
        self.device = device
        
        if model_config is None:
            model_config = torch.load(os.path.join(model_path, "model_config.pt"))
        
        # 初始化模型
        self.model = LoRADualTaskModel(model_config).to(device)
        self.model.load_pretrained(model_path)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.processor = DataProcessor()
        
        logger.info(f"预测器已加载: {model_path}")
    
    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        批量预测
        Returns:
            list of dict: [{'text': ..., 'sentiment': 'positive', 'stance': 'favor', 
                          'sentiment_conf': 0.95, 'stance_conf': 0.88}, ...]
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [self.processor.clean_text(t) for t in batch_texts]
            
            # 编码
            encodings = self.tokenizer(
                batch_texts,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # 预测
            outputs = self.model(input_ids, attention_mask)
            
            # 处理结果
            sentiment_probs = F.softmax(outputs['sentiment_logits'], dim=1)
            stance_probs = F.softmax(outputs['stance_logits'], dim=1)
            
            sentiment_preds = torch.argmax(sentiment_probs, dim=1).cpu().numpy()
            stance_preds = torch.argmax(stance_probs, dim=1).cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                results.append({
                    'text': text,
                    'sentiment': self.processor.SENTIMENT_ID2LABEL[sentiment_preds[j]],
                    'sentiment_conf': float(sentiment_probs[j][sentiment_preds[j]]),
                    'stance': self.processor.STANCE_ID2LABEL[stance_preds[j]],
                    'stance_conf': float(stance_probs[j][stance_preds[j]]),
                    'sentiment_probs': sentiment_probs[j].cpu().numpy().tolist(),
                    'stance_probs': stance_probs[j].cpu().numpy().tolist()
                })
        
        return results
    
    def predict_single(self, text: str) -> Dict:
        """单条文本预测"""
        return self.predict([text])[0]


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='LoRA双任务分类训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'demo'])
    parser.add_argument('--data_path', type=str, default=None, help='数据路径')
    parser.add_argument('--model_path', type=str, default='./output/best_model', help='模型路径')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA秩')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_stance_infer', action='store_true', help='禁用立场自动推断')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if args.mode == 'train':
        # 配置
        model_config = ModelConfig(lora_r=args.lora_r)
        train_config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output_dir
        )
        
        # 准备数据
        if args.data_path:
            if args.data_path.endswith('.json'):
                df = DataProcessor.load_from_json(args.data_path)
            else:
                df = DataProcessor.load_from_csv(
                    args.data_path, 
                    auto_infer_stance=not args.no_stance_infer
                )
        else:
            logger.warning("未提供数据路径，使用示例数据")
            df = DataProcessor.create_sample_data(2000)
        
        logger.info(f"数据样本数: {len(df)}")
        logger.info(f"情感分布:\n{df['sentiment_label'].value_counts().sort_index()}")
        logger.info(f"立场分布:\n{df['stance_label'].value_counts().sort_index()}")
        
        # 划分数据集
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df['sentiment_label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df['sentiment_label'])
        
        logger.info(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
        
        # 初始化tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        
        # 创建数据集
        train_dataset = WeiboDualDataset(
            train_df['text'].tolist(),
            train_df['sentiment_label'].tolist(),
            train_df['stance_label'].tolist(),
            tokenizer,
            model_config.max_length
        )
        val_dataset = WeiboDualDataset(
            val_df['text'].tolist(),
            val_df['sentiment_label'].tolist(),
            val_df['stance_label'].tolist(),
            tokenizer,
            model_config.max_length
        )
        test_dataset = WeiboDualDataset(
            test_df['text'].tolist(),
            test_df['sentiment_label'].tolist(),
            test_df['stance_label'].tolist(),
            tokenizer,
            model_config.max_length
        )
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size)
        
        # 初始化模型
        model = LoRADualTaskModel(model_config)
        
        # 训练
        trainer = DualTaskTrainer(model, train_config, device)
        trainer.train(train_loader, val_loader, test_loader)
        
    elif args.mode == 'predict':
        # 加载模型进行预测
        predictor = DualTaskPredictor(args.model_path, device=device)
        
        # 示例预测
        test_texts = [
            "这个产品真的太棒了，完全超出预期！强烈支持！",
            "一般般吧，没什么特别的感受，保持中立。",
            "太差了，完全不值得购买，坚决反对这种做法。",
            "虽然价格有点贵，但是质量很好，还是支持的。"
        ]
        
        results = predictor.predict(test_texts)
        
        print("\n预测结果:")
        for res in results:
            print(f"\n文本: {res['text']}")
            print(f"情感: {res['sentiment']} (置信度: {res['sentiment_conf']:.3f})")
            print(f"立场: {res['stance']} (置信度: {res['stance_conf']:.3f})")
            print(f"情感概率分布: {res['sentiment_probs']}")
            print(f"立场概率分布: {res['stance_probs']}")
    
    elif args.mode == 'demo':
        # 快速演示：训练+预测
        logger.info("运行演示模式...")
        
        # 创建示例数据
        df = DataProcessor.create_sample_data(1000)
        
        # 保存临时数据
        os.makedirs('./temp', exist_ok=True)
        temp_data_path = './temp/demo_data.json'
        df.to_json(temp_data_path, orient='records', force_ascii=False)
        
        # 训练
        args.data_path = temp_data_path
        args.epochs = 3
        args.batch_size = 16
        main()
        
        # 预测
        args.mode = 'predict'
        main()


if __name__ == "__main__":
    main()