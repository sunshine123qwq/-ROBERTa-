# fix_stance.py
import pandas as pd
import re

def extract_features(text):
    """提取立场特征"""
    text = str(text)
    
    # 支持词
    support = ['支持', '赞同', '同意', '赞', '看好', '拥护', '坚定', '必须', 
               '肯定', '正确', '好', '棒', '优秀', '喜欢', '爱', '推荐', '值得']
    # 反对词
    against = ['反对', '抗议', '拒绝', '否定', '错误', '差', '坏', '垃圾', 
               '讨厌', '恶心', '失望', '愤怒', '不满', '抵制', '不建议', '千万别']
    
    s = sum(1 for w in support if w in text)
    a = sum(1 for w in against if w in text)
    return s, a

def infer_stance(text, sentiment):
    s, a = extract_features(text)
    
    if sentiment == 2:  # 正面
        if a > s: return 0  # 反对（讽刺）
        elif s > 0: return 2  # 支持
        else: return 1  # 中立
    elif sentiment == 0:  # 负面
        if s > a: return 2  # 支持（批评但支持）
        elif a > 0: return 0  # 反对
        else: return 1  # 中立
    else:
        return 1

df = pd.read_csv('./weibo_for_pipeline.csv', encoding='utf-8')
df['stance'] = df.apply(lambda x: infer_stance(x['text'], x['sentiment']), axis=1)

print("立场分布:")
print(df['stance'].value_counts().sort_index())

df.to_csv('./weibo_fixed.csv', index=False, encoding='utf-8')
print("已保存为 weibo_fixed.csv")