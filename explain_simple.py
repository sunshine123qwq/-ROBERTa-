#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最简可解释性分析脚本 - 修复字体问题（最终版）
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import argparse
import torch
import numpy as np

# ========== 字体设置（修复版 - 关键修改）==========
import matplotlib
matplotlib.use('Agg')  # 先设置后端，避免显示问题
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体 - 修复版"""
    # 常见中文字体列表（按优先级排序）
    chinese_font_names = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 
        'FangSong', 'KaiTi', 'STHeiti', 'STSong',
        'Adobe Heiti Std', 'Adobe Song Std',
        'Noto Sans CJK SC', 'Source Han Sans SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'AR PL UMing', 'AR PL UKai'
    ]
    
    # 获取系统所有字体
    system_fonts = fm.findSystemFonts()
    
    # 查找中文字体
    found_font = None
    found_name = None
    
    for font_path in system_fonts:
        try:
            prop = fm.FontProperties(fname=font_path)
            font_name = prop.get_name()
            
            # 检查字体名是否匹配
            for cf in chinese_font_names:
                if cf.lower() in font_name.lower() or cf.lower() in os.path.basename(font_path).lower():
                    found_font = font_path
                    found_name = font_name
                    print(f"✓ 找到中文字体: {font_name} ({font_path})")
                    break
            if found_font:
                break
        except:
            continue
    
    if found_font:
        # 关键修复1：使用 FontProperties 对象而不是字体名
        font_prop = fm.FontProperties(fname=found_font)
        
        # 关键修复2：设置全局字体参数
        plt.rcParams['font.sans-serif'] = [found_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 关键修复3：设置默认字体属性
        plt.rcParams['font.family'] = 'sans-serif'
        
        return font_prop
    else:
        print("⚠️ 未找到中文字体，中文将显示为方框或乱码")
        print("  建议安装: SimHei 或 Microsoft YaHei")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return None

# 初始化字体
chinese_font = setup_chinese_font()
# =====================================

from transformers import AutoTokenizer
from lora_dual_task import LoRADualTaskModel, ModelConfig


def load_model(model_path, device='cuda'):
    """加载模型"""
    print(f"\n加载模型: {model_path}")
    
    config_path = os.path.join(model_path, "model_config.pt")
    model_config = torch.load(config_path, map_location='cpu')
    
    model = LoRADualTaskModel(model_config)
    model.load_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    print(f"✓ 模型加载完成")
    return model, tokenizer


def analyze_text_simple(model, tokenizer, text, task='sentiment', save_dir='./explanations'):
    """
    简化版可解释性分析
    """
    import jieba
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    
    print(f"\n{'='*70}")
    print(f"文本: {text}")
    print(f"{'='*70}")
    
    # 1. 预测（无梯度）
    encoding = tokenizer(text, return_tensors='pt', max_length=128, 
                        truncation=True, padding='max_length')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        sentiment_logits = outputs['sentiment_logits']
        stance_logits = outputs['stance_logits']
        
        sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
        stance_pred = torch.argmax(stance_logits, dim=1).item()
        sentiment_proba = torch.softmax(sentiment_logits, dim=1)[0].cpu().numpy()
        stance_proba = torch.softmax(stance_logits, dim=1)[0].cpu().numpy()
    
    # 标签映射
    sentiment_labels = {0: '负面', 1: '中性', 2: '正面'}
    stance_labels = {0: '反对', 1: '中立', 2: '支持'}
    
    print(f"\n【预测结果】")
    print(f"  情感: {sentiment_labels[sentiment_pred]} (置信度: {sentiment_proba[sentiment_pred]:.3f})")
    print(f"         分布: [负:{sentiment_proba[0]:.3f}, 中:{sentiment_proba[1]:.3f}, 正:{sentiment_proba[2]:.3f}]")
    print(f"  立场: {stance_labels[stance_pred]} (置信度: {stance_proba[stance_pred]:.3f})")
    print(f"         分布: [反:{stance_proba[0]:.3f}, 中:{stance_proba[1]:.3f}, 支:{stance_proba[2]:.3f}]")
    
    # 2. 注意力分析（替代梯度分析，避免梯度问题）
    print(f"\n【注意力分析 - 词重要性】")
    
    with torch.no_grad():
        # 获取attention
        base_outputs = model.lora_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # 使用最后一层attention的CLS token注意力
        last_attention = base_outputs.attentions[-1]  # (batch, heads, seq, seq)
        # 平均所有头，取CLS对其他token的注意力
        cls_attention = last_attention[0, :, 0, :].mean(dim=0).cpu().numpy()
        
        # 归一化
        token_importance = cls_attention / (cls_attention.max() + 1e-10)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # 过滤并显示
    valid_pairs = [(t, imp) for t, imp in zip(tokens, token_importance) 
                   if t not in ['', '[CLS]', '[SEP]', '']]
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Top-10 重要token:")
    for i, (token, imp) in enumerate(valid_pairs[:10], 1):
        print(f"    {i:2d}. {token:15s} | 注意力: {imp:.4f}")
    
    # 3. jieba分词 + 聚合
    print(f"\n【词级分析 - jieba分词】")
    words = list(jieba.cut(text))
    
    # 聚合到词级别
    word_scores = []
    for word in words:
        if not word.strip() or word in ['，', '。', '！', '？', '、', '；', '：', ' ', '\n']:
            continue
            
        # 找到相关token
        related_scores = []
        for token, score in valid_pairs:
            clean_token = token.replace('##', '').replace('', '')
            if clean_token in word or word in clean_token:
                related_scores.append(score)
        
        if related_scores:
            avg_score = np.mean(related_scores)
            word_scores.append((word, avg_score))
    
    # 排序并去重
    word_scores.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_word_scores = []
    for word, score in word_scores:
        if word not in seen:
            seen.add(word)
            unique_word_scores.append((word, score))
    
    print(f"  Top-10 重要词:")
    for i, (word, score) in enumerate(unique_word_scores[:10], 1):
        print(f"    {i:2d}. {word:15s} | 注意力: {score:.4f}")
    
    # 4. 可视化
    task_name = "情感" if task == 'sentiment' else "立场"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 4.1 预测概率
    ax = axes[0, 0]
    x = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x - width/2, sentiment_proba, width, label='Sentiment', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, stance_proba, width, label='Stance', color='#e74c3c', alpha=0.8)
    
    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Neg/Oppose', 'Neutral', 'Pos/Support'])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 4.2 Token重要性
    ax = axes[0, 1]
    top_tokens = valid_pairs[:15]
    tokens_list = [t[0].replace('##', '') for t in top_tokens]
    scores_list = [t[1] for t in top_tokens]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores_list)))
    bars = ax.barh(range(len(tokens_list)), scores_list, color=colors)
    ax.set_yticks(range(len(tokens_list)))
    ax.set_yticklabels(tokens_list, fontsize=9)
    ax.set_xlabel('Attention Weight')
    ax.set_title(f'Token Importance ({task_name})')
    ax.invert_yaxis()
    
    # 4.3 词重要性 - 关键修复：使用 fontproperties 参数
    ax = axes[1, 0]
    top_words = unique_word_scores[:15]
    words_list = [w[0] for w in top_words]
    word_scores_list = [w[1] for w in top_words]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(word_scores_list)))
    ax.barh(range(len(words_list)), word_scores_list, color=colors)
    ax.set_yticks(range(len(words_list)))
    
    # 关键修复4：使用 fontproperties 参数设置中文字体
    if chinese_font:
        # 为每个标签单独设置字体
        labels = []
        for word in words_list:
            labels.append(word)
        ax.set_yticklabels(labels, fontsize=10, fontproperties=chinese_font)
    else:
        ax.set_yticklabels([f"Word_{i}" for i in range(len(words_list))], fontsize=10)
    
    ax.set_xlabel('Aggregated Attention')
    ax.set_title('Word-Level Importance (Jieba Segmentation)', fontproperties=chinese_font if chinese_font else None)
    ax.invert_yaxis()
    
    # 4.4 总结
    ax = axes[1, 1]
    ax.axis('off')
    
    # 构建总结文本
    summary = f"""Analysis Summary ({task_name} Task)
    
Prediction:
  Sentiment: {sentiment_labels[sentiment_pred]} ({sentiment_proba[sentiment_pred]:.3f})
  Stance: {stance_labels[stance_pred]} ({stance_proba[stance_pred]:.3f})

Top Evidence Words:
"""
    for i, (w, s) in enumerate(unique_word_scores[:5], 1):
        summary += f"{i}. {w} ({s:.3f})\n"
    
    # 关键修复5：在 text 中也使用 fontproperties
    text_props = {'family': 'monospace'}
    if chinese_font:
        text_props['fontproperties'] = chinese_font
    
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            **text_props)
    
    # 关键修复6：调整布局，确保中文显示完整
    plt.tight_layout()
    
    # 关键修复7：保存时使用更高的 DPI 和 bbox_inches
    save_path = os.path.join(save_dir, f"analysis_{task}_{abs(hash(text)) % 10000}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"\n✓ 可视化已保存: {save_path}")
    
    # 显示（如果系统支持）
    try:
        plt.show()
    except:
        pass
    
    plt.close(fig)  # 关键修复8：关闭图形，释放内存
    
    return {
        'text': text,
        'sentiment': sentiment_labels[sentiment_pred],
        'stance': stance_labels[stance_pred],
        'top_words': unique_word_scores[:10]
    }


def main():
    parser = argparse.ArgumentParser(description='最简可解释性分析')
    parser.add_argument('--model_path', type=str, default='./output/best_model')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--task', type=str, default='sentiment', choices=['sentiment', 'stance'])
    parser.add_argument('--save_dir', type=str, default='./explanations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载模型
    try:
        model, tokenizer = load_model(args.model_path, args.device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 分析
    if args.text:
        analyze_text_simple(model, tokenizer, args.text, args.task, args.save_dir)
    else:
        # 示例
        examples = [
            ("这个产品真的太棒了，完全超出预期！强烈支持！", "sentiment"),
            ("虽然价格有点贵，但是质量很好，还是支持的。", "stance"),
            ("一般般吧，没什么特别的感受。", "sentiment"),
            ("太差了，完全不值得购买，坚决反对这种做法。", "stance"),
            ("这个政策虽然有争议，但总体来说是正确的，我支持。", "stance"),
        ]
        
        for text, task in examples:
            try:
                analyze_text_simple(model, tokenizer, text, task, 
                                   os.path.join(args.save_dir, task))
                print("\n" + "="*70)
            except Exception as e:
                print(f"分析失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ 全部完成！结果保存至: {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())