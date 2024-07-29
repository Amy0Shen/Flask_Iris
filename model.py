from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

iris = load_iris()
X = iris.data[:, :2]  # We will use only sepal length and width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
regressor = LogisticRegression(max_iter=200)
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5.0, 3.5]]))
print(model.predict([[5.5, 2.5]]))
print(model.predict([[7.5, 3.0]]))
