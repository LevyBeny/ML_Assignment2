import SVR_gradient_boosting as svr_boosting
import numpy as np
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# slump= np.genfromtxt('./slump.csv',delimiter=',')
# X, y = shuffle(slump[:,:-1], slump[:,-1], random_state=13)
# X = X.astype(np.float32)
# offset = int(X.shape[0] * 0.7)
# X_train, y_train = X[:offset], y[:offset]
# X_test, y_test = X[offset:], y[offset:]
#
#
# params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
#           'learning_rate': 0.01, 'loss': 'ls','init':'svr'}
# clf = svr_boosting.GradientBoostingRegressor(**params)
# clf.fit(X_train, y_train)
#
# Predict= clf.predict(X_test)
# mse = mean_squared_error(y_test,Predict)
# print("MSE: %.4f" % mse)

'''forestfires'''
fire= np.genfromtxt('./forestfires.csv',delimiter=',')
X, y = shuffle(fire[:,:-1], fire[:,-1], random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.7)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls','init':'svr'}
clf = svr_boosting.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

Predict= clf.predict(X_test)
mse = mean_squared_error(y_test,Predict)
print("MSE: %.4f" % mse)