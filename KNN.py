import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report
class KNNClassification:
    def __init__(self):
        x_train = None
        y_train = None

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def predict(self, x_test, k):
        self.x_test = np.array(x_test)
        predictions = []
        for x in x_test:
            distance = np.sqrt(np.sum((self.x_train-x)**2, axis=1))
            nearest_indices = np.argsort(distance)[:k]
            nearest_label = self.y_train[nearest_indices]
            most_common = Counter(nearest_label).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

if __name__ == '__main__':
    data = datasets.load_iris()
    print(data)
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.2)
    knn = KNNClassification()
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test, 3)
    print(classification_report(y_test, predict))


