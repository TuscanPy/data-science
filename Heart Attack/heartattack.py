import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def best_tree(max_leaf_nodes):
    best_tree_list = {}
    for leaf in max_leaf_nodes:
        model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=leaf)
        model.fit(X_train, y_train)
        prediction = model.predict(X_valid)
        best_tree_list[leaf] = mean_absolute_error(y_valid, prediction)
    return min(best_tree_list, key=best_tree_list.get)


def model_score(model, X_t, y_t, y_v):
    model.fit(X_t, y_t)
    return mean_absolute_error(y_v, model.predict(X_valid))


heart_df = pd.read_csv('heart.csv')
y = heart_df['output']
features = ['age', 'sex', 'cp', 'trtbps', 'thall', 'fbs']
X = heart_df[features]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, test_size=0.2, train_size=0.8)


model_1 = DecisionTreeRegressor(random_state=0, max_leaf_nodes=33)
print(model_score(model_1, X_train, y_train, y_valid))
