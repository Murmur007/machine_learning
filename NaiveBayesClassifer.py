import numpy as np
from collections import defaultdict
import re
from sklearn .datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        """alpha：平滑系数（拉普拉斯平滑）"""
        self.alpha = alpha
        self.class_probs = {}    # 类别先验概率 P(C)
        self.word_probs = {}     # 条件概率 P(W|C)
        self.vocab = set()       # 词汇表
        self.classes = []        # 类别列表

    def preprocess(self, text):
        """文本预处理：小写化、去除非字母字符"""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        return text.split()

    def fit(self, X, y):
        """统计类别信息"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_docs = len(X)
        """计算类别先验概率 P(C)"""
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / n_docs
        """初始化词频统计字典"""
        word_counts = {c: defaultdict(int) for c in self.classes}
        class_word_totals = {c: 0 for c in self.classes}
        """统计每个类别中的词频"""
        for doc, cls in zip(X, y):
            words = self.preprocess(doc)
            for word in words:
                word_counts[cls][word] += 1
                class_word_totals[cls] += 1
                self.vocab.add(word)
        vocab_size = len(self.vocab)
        """计算条件概率 P(W|C)使用拉普拉斯平滑"""
        self.word_probs = {c: defaultdict(float) for c in self.classes}
        for c in self.classes:
            total_words = class_word_totals[c]
            for word in self.vocab:
                count = word_counts[c][word] + self.alpha
                self.word_probs[c][word] = count / (total_words + self.alpha + vocab_size)

    def predict(self, X):
        predictions = []
        for doc in X:
            words = self.preprocess(doc)
            max_prob = -np.inf
            best_class = None
            """对每个类别计算后验概率"""
            for c in self.classes:
                log_prob = np.log(self.class_probs[c])    # log P(C)
                """累加所有单词的log P(W|C)"""
                for word in words:
                    if word in self.word_probs[c]:
                        log_prob +=np.log(self.word_probs[c][word])
                    else:
                        """处理未登录词（使用平滑）"""
                        log_prob += np.log(self.alpha / (sum(len(self.word_probs[cls]) for cls in self.classes) + self.alpha * len(self.vocab)))
                if log_prob > max_prob:
                    max_prob = log_prob
                    best_class = c
            predictions.append(best_class)
        return predictions

if __name__ == '__main__':
    categories = ['sci.med', 'comp.graphics', 'talk.politics.guns']
    newsgroup = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    X, y = newsgroup.data, newsgroup.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"测试集准确率：{accuracy:.4f}")

    # 查看各类别预测示例
    for i in range(5):
        print(f"\n真实类别: {categories[y_test[i]]}")
        print(f"预测类别: {categories[y_pred[i]]}")
        print("文本片段:", X_test[i][:200], "...")