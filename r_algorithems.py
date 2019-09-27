# KNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

K_value = 10
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train)
testt=neigh.predict(X_test)
scoree=accuracy_score(y_test, testt)
print("KNN testing accuracy=",scoree*100)
#SVM
from sklearn.svm import SVC 

svmclassifier = SVC(kernel='linear')  
svmclassifier.fit(X_train, y_train)  
svm_y_pred = svmclassifier.predict(X_test)
svm_score = accuracy_score(y_test, svm_y_pred)
print("SVM testing accuracy=",svm_score*100)
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps

for i in range(15,21):
    Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=i)
    Tree_model.fit(X_train, y_train)
    tree_pred = Tree_model.predict(X_test)
    tree_score = accuracy_score(y_test, tree_pred)
    print("Decision testing accuracy (depth=",i,")=",tree_score*100)

     #print(Tree_model.predict_proba(np.array([X])))
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=500, max_depth = 15 ,random_state=70)  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred_round = [round(x) for x in y_pred]

y_score = accuracy_score(y_test, y_pred_round)
print("Random forest testing accuracy=",y_score*100)
# Gradient boosting
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

params = {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
GD_score = accuracy_score(y_test, [round(x) for x in clf.predict(X_test)])
print("Gradient Boosting testing accuracy=",GD_score*100)
