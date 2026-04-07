from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
# Load the Iris dataset
data = load_iris()
#print(data)
X = data.data
y = data.target
print(X)
# 1. Split the dataset into training and testing sets , data to detect overfitting/ underfitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# testing different values of k
k_values = range(1, 31)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    #using cross validation to evaluate the model's performance for each k
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())
#visulize the results
plt.figure(figsize=(8,5))
plt.plot(k_values, cv_scores, marker='*',color='pink')
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding the Optimal k (bias-variance tradeoff)')
plt.grid(True)
plt.show()