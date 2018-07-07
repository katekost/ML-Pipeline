import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics
from sklearn.utils import shuffle
import statistics as st

print(os.getcwd())

def MetricsForMulticlass(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    matrix = metrics.confusion_matrix(y_true, y_pred)
    print("Quality metrics for this model: ")
    print(" Accuracy:  ", round(accuracy * 100), "%")
    print(" Kappa:     ", round(kappa, 3))
    print("Confusion matrix:")
    print(matrix)

dataset_top =  pd.read_csv('multi_full_731.csv', sep=";", decimal=",")
print("Number of rows: ", dataset_top.shape[0]) #4761
print("Number of features: ", dataset_top.shape[1]) #731

dop =  pd.read_csv('mcc_groups_trans_num.csv', sep=";", decimal=",")
print("Number of rows: ", dop.shape[0]) #7517
print("Number of features: ", dop.shape[1]) #11

dataset = pd.merge(dataset_top, dop, how="inner", on="id_resp")
print("Number of rows: ", dataset.shape[0]) #4761
print("Number of features: ", dataset.shape[1]) #741

def recode(series):
    if series == 'Macro3':
        return 1
    elif series == 'Macro8':
        return 2
    else:
        return 0

dataset['target38'] = dataset.target.apply(recode)
group = pd.DataFrame(dataset.groupby('target38').count())
print(dataset.columns)
print(dataset.groupby('target38').groups.keys())
print("Nulls in dataset: ", dataset.isnull().values.any())

dataset=dataset.drop(columns=["target"])

dataset = shuffle(dataset)
dataset_n = np.array(dataset)

n_rows = dataset.shape[0] # 4761
n_test = round((n_rows*30)/100, 0)
test = dataset_n[0:int(n_test)-1, :]

for i in range(0, int(n_test)-1):
    dataset_n = np.delete(dataset_n, i, 0)

dataset_n.shape[0]

part_first = dataset_n[0:1111, :]
part_second = dataset_n[1112:2222, :]
part_third = dataset_n[2223:3333, :]
print("Number of rows and columns in first part is: ", part_first.shape)
print("Number of rows and columns in second part is: ", part_second.shape)
print("Number of rows and columns in third part is: ", part_third.shape)

data_log = np.concatenate((part_first, part_second), axis=0)
data_gbm = np.concatenate((part_first, part_third), axis=0)
data_forest = np.concatenate((part_second, part_third), axis=0)

test_upd = test[:, 1:739]
#----------------------------------------------------------------------------
# LOGISTIC REGRESSION

def improvement(frame, acc):
    sum = frame.iloc[0, 0] + frame.iloc[1, 0] + frame.iloc[2, 0]
    diag_sum = (frame.iloc[0, 0]/sum)**2  + (frame.iloc[1, 0]/sum)**2 + (frame.iloc[2, 0]/sum)**2
    return (acc - diag_sum) * 100

n_dataset_train = data_log[:, 1:739]
n_dataset_target = data_log[:, 740]

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

matrix48 = metrics.confusion_matrix(y_test, pred)
MetricsForMulticlass(y_test, pred)

with open("top_log_nosel_newfeat_38_blend", "wb") as f:
    pickle.dump(lm, f, pickle.HIGHEST_PROTOCOL)

accuracy = metrics.accuracy_score(y_test, pred)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))

test_upd = test[:, 1:739]
pred_prob_log = lm.predict_proba(test_upd)


#----------------------------------------------------------------------------
# GBM

n_dataset_train = data_gbm[:, 1:739]
n_dataset_target = data_gbm[:, 740]

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

g_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y = np.argmax(g_pred, axis=1)
MetricsForMulticlass(y_test, y)

with open("top_gbm_no_sel_new_feat_38_blend", "wb") as f:
    pickle.dump(gbm, f, pickle.HIGHEST_PROTOCOL)

accuracy = metrics.accuracy_score(y_test, y)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))

gbm_pred = gbm.predict(test_upd)

#----------------------------------------------------------------------------
# RANDOM FOREST

n_dataset_train = data_forest[:, 1:739]
n_dataset_target = data_forest[:, 740]

X_train, X_test, y_train, y_test = train_test_split(
    n_dataset_train, n_dataset_target, test_size=0.3, random_state=42)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

forest = rfc(max_depth=5, n_estimators=50, random_state=42, n_jobs=-1)
forest = forest.fit(X_train, y_train)
pred_f = forest.predict(X_test)
pred_prob = forest.predict_proba(X_test)

train_score = forest.score(X_train, y_train)
test_score = forest.score(X_test, y_test)

print("Train Accuracy: ", train_score)
print("Test Accuracy: ", test_score)

matrix38 = metrics.confusion_matrix(y_test, pred)
MetricsForMulticlass(y_test, pred)

with open("top_rforest_no_sel_newfeatures_38_blend", "wb") as f:
    pickle.dump(forest, f, pickle.HIGHEST_PROTOCOL)

accuracy = metrics.accuracy_score(y_test, pred_f)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))


predictors_for_top = pd.DataFrame(dataset.columns)

pred_proba_forest = forest.predict_proba(test_upd)

#----------------------------------------------------------------------------
# AVERAGING

forest_ser = pd.DataFrame(pred_proba_forest, columns=['forest0', 'forest1', 'forest2'])
gbm_ser = pd.DataFrame(gbm_pred, columns=['gbm0', 'gbm1', 'gbm2'])
log_ser = pd.DataFrame(pred_prob_log, columns=['log0', 'log1', 'log2'])

avg_prob0 = pd.concat([forest_ser, gbm_ser], axis=1)
avg_prob0 = pd.concat([avg_prob0, log_ser], axis=1)

avg_prob = avg_prob0[['log0', 'gbm0', 'forest0', 'log1', 'gbm1', 'forest1', 'log2', 'gbm2', 'forest2']]

avg_prob['mean0'] = avg_prob[['log0', 'gbm0', 'forest0']].mean(axis=1)
avg_prob['mean1'] = avg_prob[['log1', 'gbm1', 'forest1']].mean(axis=1)
avg_prob['mean2'] = avg_prob[['log2', 'gbm2', 'forest2']].mean(axis=1)
avg_prob.__delitem__('mean00')
avg_prob["ans_prob"] = np.nan

for i in range(0, avg_prob.shape[0]):
    if max(avg_prob.iloc[i, 9], avg_prob.iloc[i, 10], avg_prob.iloc[i, 11]) == avg_prob.iloc[i, 9]:
        avg_prob.iloc[i, 12] = 0
    elif max(avg_prob.iloc[i, 9], avg_prob.iloc[i, 10], avg_prob.iloc[i, 11]) == avg_prob.iloc[i, 10]:
        avg_prob.iloc[i, 12] = 1
    else:
        avg_prob.iloc[i, 12] = 2

avg_prob.ans_prob = avg_prob.ans_prob.astype('int')
print(avg_prob.groupby('ans_prob').groups.keys())
print(avg_prob.groupby('ans_prob').count())

pred = np.array(avg_prob.ans_prob)
accuracy = metrics.accuracy_score(test[:, 740], pred)
imp = improvement(group, accuracy)
print("Accuracy: ", round(accuracy, 2))
print("Improvement: ", round(imp, 2))

#----------------------------------------------------------------------------

dataset_top =  pd.read_csv('multi_full_731.csv', sep=";", decimal=",")
print("Number of rows: ", dataset_top.shape[0])
print("Number of features: ", dataset_top.shape[1])

dop = pd.read_csv("D:/BI/mltscn_combine/Diff_by_client_idresp.csv", sep=";", decimal=",")
print("Number of rows: ", dop.shape[0])
print("Number of features: ", dop.shape[1])
dop.__delitem__('Customer_Id')


dataset_test = pd.merge(dataset_top, dop, how="inner", on="id_resp")
print("Number of rows: ", dataset_test.shape[0])
print("Number of features: ", dataset_test.shape[1])



