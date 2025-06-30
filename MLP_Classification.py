import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class MLP_Classificiation:
    def __init__(self, hidden_layers=(100,50), activation='relu', learning_rate=0.01, epochs=100):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = OrderedDict()
        self.bias = OrderedDict()

    def _init_weight(self, input_dim, output_dim):
        """初始化权重和偏置"""
        layers_dim = [input_dim] + list(self.hidden_layers) + [output_dim]

        for i in range(1, len(layers_dim)):
            """生成形状为 (layers_dim[i - 1], layers_dim[i]) 的随机矩阵，值在 [0,1) 均匀分布"""
            self.weights[f'W{i}'] = np.random.rand(layers_dim[i - 1], layers_dim[i])*0.5
            self.bias[f'b{i}'] = np.zeros((1, layers_dim[i]))

    def _activation_fn(self, z):
        """激活函数"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-z))
        elif self.activation == 'tanh':
            return (1-np.exp(-2*z)) / (1+np.exp(-2*z))
        else:
            raise ValueError('Activation function not supported')

    def _activation_derivative(self, a):
        """激活函数导数"""
        if self.activation == 'relu':
            return (a>0).astype(float)
        elif self.activation == 'sigmoid':
            return a*(1-a)
        elif self.activation == 'tanh':
            return 1-np.square(a)

    def _softmax(self, z):
        """Softmax函数用于多分类输出层"""
        # axis=1按行操作，exp_z是矩阵中的一个元素，keepdims将按行计算后的矩阵再回复成原来的行维度与矩阵元素进行计算
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        self.cache = {'A0': X}
        for i in range(1, len(self.weights)+1):
            z = np.dot(self.cache[f'A{i-1}'], self.weights[f'W{i}']) + self.bias[f'b{i}']
            # 每一层经过线性计算后的结果
            self.cache[f'Z{i}'] = z
            if i == len(self.weights):
                """输出层用softmax"""
                a = self._softmax(z)
            else:
                """隐藏层用指定激活函数"""
                a = self._activation_fn(z)
            # 每一层经过当前隐藏层后的激活输出
            self.cache[f'A{i}'] = a
            # 返回输出层的计算值
        return self.cache[f'A{len(self.weights)}']

    def backward(self, X, y):
        """反向传播"""
        m = X.shape[0]
        grads = {}
        L = len(self.weights)

        # 输出层梯度
        dZ = self.cache[f'A{L}'] - y
        grads[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZ) / m
        grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m

        # 隐藏层梯度
        for l in reversed(range(1, L)):
            dA = np.dot(dZ, self.weights[f'W{l+1}'].T)
            dZ = dA * self._activation_derivative(self.cache[f'A{l}'])
            grads[f'dW{l}'] = np.dot(self.cache[f'A{l-1}'].T, dZ) / m
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    def fit(self, X, y):
        """训练模型"""
        y_onehot = np.eye(len(np.unique(y)))[y]
        self._init_weight(X.shape[1], y_onehot.shape[1])

        for epoch in range(self.epochs):
            # 前向传播
            y_pred = self.forward(X)

            # 计算损失（交叉熵）
            loss = -np.mean(np.log(y_pred[np.arange(len(y)), y]))

            # 反向传播
            grads = self.backward(X, y_onehot)

            # 更多参数
            for i in range(1, len(self.weights)+1):
                self.weights[f'W{i}'] -= self.lr * grads[f'dW{i}']
                self.bias[f'b{i}'] -= self.lr * grads[f'db{i}']

            if epoch % 10 == 0:
                print(f'Epoch {epoch}， Loss：{loss:.4f}')

    def predict(self, X):
        """预测类别"""
        proba = self.forward(X)
        return np.argmax(proba, axis=1)

if __name__ == '__main__':
    digits = load_digits()
    X_data = digits.data
    y_data = digits.target

    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

    mlp = MLP_Classificiation(hidden_layers=(100,50), activation='relu', learning_rate=0.01, epochs=3000)
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    print(classification_report(y_test, predictions))