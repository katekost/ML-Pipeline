import pickle
import os
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

print("Current working directory: ", os.getcwd())
#----------------------------------------------------------------------------
# WARNING: IF PRED VECTOR IS A VECTOR WITH CLASSES LABELS

def DataSplitting(dataset, test_share):
    np.random.seed(42)
    dataset = shuffle(dataset)
    dataset_n = np.array(dataset)
    n_dataset_train = dataset_n[:, 1:dataset_n.shape[1] - 2]
    n_dataset_target = dataset_n[:, dataset_n.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(
        n_dataset_train, n_dataset_target, test_size=test_share, random_state=42)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    return X_train, X_test, y_train, y_test

def FittingModel(estimator, X_train, y_train):
    model = estimator.fit(X_train, y_train)
    return model

def MetricsForMulticlass(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    matrix = metrics.confusion_matrix(y_true, y_pred)
    print("Quality metrics for this model: ")
    print(" Accuracy:  ", round(accuracy * 100), "%")
    print(" Kappa:     ", round(kappa, 3))
    print("Confusion matrix:")
    print(matrix)

def Improvement(frame, acc):
    sum = frame.iloc[0, 0] + frame.iloc[1, 0] + frame.iloc[2, 0]
    diag_sum = (frame.iloc[0, 0]/sum)**2  + (frame.iloc[1, 0]/sum)**2 + (frame.iloc[2, 0]/sum)**2
    return (acc - diag_sum) * 100

def PrintEvaluations(model, X_train, y_train, X_test, y_test):
    group = pd.DataFrame(dataset.groupby('target38').count())
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    pred = model.predict(X_test)
    confus_matrix = metrics.confusion_matrix(y_test, pred)
    MetricsForMulticlass(y_test, pred)
    accuracy = metrics.accuracy_score(y_test, pred)
    imp = Improvement(group, accuracy)
    print("Train Accuracy: ", train_score)
    print("Test Accuracy: ", test_score)
    print("Confusion Matrix: ", confus_matrix)
    print("Accuracy: ", round(accuracy, 2))
    print("Improvement: ", round(imp, 2))

def SaveModel(model, name_of_model):
    with open(name_of_model, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def LoadModel(model, name_of_model):
    with open(name_of_model, "rb") as f:
        pickle.load(model, f, pickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------------
dataset = pd.read_csv("D:\BI\mltscn_combine_0.1_data\dataset_all_features_for_forest.csv")
dataset = dataset.drop(columns=['Unnamed: 0'])
#----------------------------------------------------------------------------
# RANDOM FOREST

forest = rfc(max_depth=5, n_estimators=500, random_state=42, n_jobs=-1)
X_train, X_test, y_train, y_test = DataSplitting(dataset, 0.3)
forest = FittingModel(forest, X_train, y_train)
PrintEvaluations(forest, X_train, y_train, X_test, y_test)
SaveModel(forest, "rforest38_all_new_feat-s") # 65% 16.85%

# SELECT MOST IMPORTANT FEATURES

imprv = pd.DataFrame(forest.feature_importances_)

index = []
imp_lev = 0.001

for i in range(0, imprv.shape[0]-1):
	if imprv.iloc[i, 0]>=imp_lev:
		index.append(i)

print(len(index))

with open("most_important_features.txt", "w") as txt_file:
    for i in index:
        txt_file.write("%s\n" % i)

np.random.seed(42)
dataset = shuffle(dataset)
dataset_n = np.array(dataset)
n_dataset_train = dataset_n[:, index]
n_dataset_target = dataset_n[:, dataset_n.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(
    n_dataset_train, n_dataset_target, test_size=0.3, random_state=42)

forest = FittingModel(forest, X_train, y_train)
PrintEvaluations(forest, X_train, y_train, X_test, y_test)


#----------------------------------------------------------------------------
dataset = pd.read_csv("D:\BI\mltscn_combine_0.1_data\dataset_all_features_for_regression.csv")
dataset = dataset.drop(columns=['Unnamed: 0'])
dataset = dataset.fillna(0)
#----------------------------------------------------------------------------
# LOGISTIC REGRESSION

logMod = LogisticRegression()
X_train, X_test, y_train, y_test = DataSplitting(dataset, 0.3)
logMod = FittingModel(logMod, X_train, y_train)
PrintEvaluations(logMod, X_train, y_train, X_test, y_test)

#----------------------------------------------------------------------------
# SIMPLE NEURAL NETWORK

mlp = MLPClassifier(solver='lbfgs', random_state=42, activation='tanh', hidden_layer_sizes=[10, 10], alpha=0.01)
X_train, X_test, y_train, y_test = DataSplitting(dataset, 0.3)
mpl = FittingModel(mlp, X_train, y_train)
PrintEvaluations(mpl, X_train, y_train, X_test, y_test)
SaveModel(mlp, "mlp_all_new_feat-s")

#-----------------------------------------------------------------------------
# GRADIENT BOOSTING

n_dataset_train = dataset_n[:, 1:dataset_n.shape[1]-2]
n_dataset_target = dataset_n[:, dataset_n.shape[1]-1]

X_train, X_test, y_train, y_test = train_test_split(
    n_dataset_train, n_dataset_target, test_size=0.3, random_state=42)

lgbm_train = lgbm.Dataset(X_train, y_train)
lgbm_eval = lgbm.Dataset(X_test, y_test, reference=lgbm_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': {'multi_logloss', 'multi_error'},
    'is_unbalance': True,
    'num_class': 3,
    'num_leaves': 30,
    'learning_rate': 0.05,
    'min_data_in_leaf': 5
}

gbm = lgbm.train(params,
                 lgbm_train,
                 valid_sets=lgbm_eval
)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

test = pd.DataFrame(y_pred)
test['class_pred'] = np.nan

for i in range(0, test.shape[0]):
    if max(test.iloc[i, 0], test.iloc[i, 1], test.iloc[i, 2]) == test.iloc[i, 0]:
        test.iloc[i, 3] = 0
    elif max(test.iloc[i, 0], test.iloc[i, 1], test.iloc[i, 2]) == test.iloc[i, 2]:
        test.iloc[i, 3] = 1
    else:
        test.iloc[i, 3] = 2

test.class_pred = test.class_pred.astype('int')
test['class_true'] = y_test
test.to_csv("test.csv")
test['check']=np.nan

for i in range(0, test.shape[0]):
	if test.iloc[i, 3]==test.iloc[i, 4]:
		test.iloc[i, 5]=1
	else:
		test.iloc[i, 5]=0

test.check = test.check.astype('int')
accuracy = test.check.sum()/(test.shape[0]-1)

imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))

# RF WITH IMPORTANT FEATURES

n_dataset_train = dataset_n[:, index]
n_dataset_target = dataset_n[:, dataset_n.shape[1]-1]

X_train, X_test, y_train, y_test = train_test_split(
    n_dataset_train, n_dataset_target, test_size=0.3, random_state=42)

lgbm_train = lgbm.Dataset(X_train, y_train)
lgbm_eval = lgbm.Dataset(X_test, y_test, reference=lgbm_train)

gbm = lgbm.train(params,
                 lgbm_train,
                 valid_sets=lgbm_eval
)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

test = pd.DataFrame(y_pred)
test['class_pred'] = np.nan

for i in range(0, test.shape[0]):
    if max(test.iloc[i, 0], test.iloc[i, 1], test.iloc[i, 2]) == test.iloc[i, 0]:
        test.iloc[i, 3] = 0
    elif max(test.iloc[i, 0], test.iloc[i, 1], test.iloc[i, 2]) == test.iloc[i, 1]:
        test.iloc[i, 3] = 1
    else:
        test.iloc[i, 3] = 2
test.class_pred = test.class_pred.astype('int')
test['class_true'] = y_test
#test.to_csv("test.csv")
test['check']=np.nan

for i in range(0, test.shape[0]):
	if test.iloc[i, 3]==test.iloc[i, 4]:
		test.iloc[i, 5]=1
	else:
		test.iloc[i, 5]=0

test.check = test.check.astype('int')
accuracy = test.check.sum()/(test.shape[0]-1)

imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))


#--------------------------------------------------------------------------

dataset.isnull().sum()
null = pd.DataFrame(dataset.isnull().sum())

dataset['target38copy'] = dataset.target38
dataset=dataset.drop(columns=["target38"])
dataset.shape

dataset_c = dataset.fillna(0)
print("Nulls in dataset: ", dataset_c.isnull().values.any())
dataset_n = np.array(dataset_c)

n_dataset_train = dataset_n[:, 1:dataset_n.shape[1]-2]
n_dataset_target = dataset_n[:, dataset_n.shape[1]-1]

X_train, X_test, y_train, y_test = train_test_split(
    n_dataset_train, n_dataset_target, test_size=0.3, random_state=42)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

logMod = LogisticRegression()
lm = logMod.fit(X_train, y_train)
pred = lm.predict(X_test)
train_score = lm.score(X_train, y_train)
test_score = lm.score(X_test, y_test)
print("Train Accuracy: ", train_score)
print("Test Accuracy: ", test_score)
accuracy = metrics.accuracy_score(y_test, pred)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2)) #0.62
print("Improvement: ", round(imp, 2)) #14.3

matrix48 = metrics.confusion_matrix(y_test, pred)
MetricsForMulticlass(y_test, pred)

# WITH CENTER SCALE

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

logMod = LogisticRegression()
lm = logMod.fit(X_train, y_train)
pred = lm.predict(X_test)

train_score = lm.score(X_train, y_train)
test_score = lm.score(X_test, y_test)

print("Train Accuracy: ", train_score)
print("Test Accuracy: ", test_score)

matrix48 = metrics.confusion_matrix(y_test, pred)
MetricsForMulticlass(y_test, pred)

with open("top_log_nosel_newfeat_38_blend", "wb") as f:
    pickle.dump(lm, f, pickle.HIGHEST_PROTOCOL)

accuracy = metrics.accuracy_score(y_test, pred)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2)) #0.62
print("Improvement: ", round(imp, 2)) #14.3

#----------------------------------------------------------------------------
# WITH MEAN ENCODING

print("Nulls in dataset: ", dataset_c.isnull().values.any())
dataset_n = np.array(dataset_c)

def encoding(series):
    if series == 0:
        return 0.6
    elif series == 1:
        return 0.1
    else:
        return 0.2

dataset_c['target_encoding'] = dataset_c.target38copy.apply(encoding)
dataset_c['target38copy2'] = dataset_c.target38copy
dataset_c=dataset_c.drop(columns=["target38copy"])
dataset_n = np.array(dataset_c)

n_dataset_train = dataset_n[:, 1:dataset_n.shape[1]-2]
n_dataset_target = dataset_n[:, dataset_n.shape[1]-1]

X_train, X_test, y_train, y_test = train_test_split(
    n_dataset_train, n_dataset_target, test_size=0.3, random_state=42)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

logMod = LogisticRegression()

lm = logMod.fit(X_train, y_train)
pred = lm.predict(X_test)

train_score = lm.score(X_train, y_train)
test_score = lm.score(X_test, y_test)
print("Train Accuracy: ", train_score)
print("Test Accuracy: ", test_score)

MetricsForMulticlass(y_test, pred)

accuracy = metrics.accuracy_score(y_test, pred)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2)) #0.62
print("Improvement: ", round(imp, 2)) #14.3

#----------------------------------------------------------------------------
# RECURSIVE FEATURES ELIMINATION

from sklearn.feature_selection import RFE
estimator = rfc(max_depth=4, n_estimators=500, random_state=42, n_jobs=-1)
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X_train, y_train)

with open("rfe_new_features38", "wb") as f:
    pickle.dump(selector, f, pickle.HIGHEST_PROTOCOL)

#selector.support_
#selector.ranking_

mod = selector.fit(X_train, y_train)
prediction = selector.predict(X_test)

accuracy = metrics.accuracy_score(y_test, pred)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))
