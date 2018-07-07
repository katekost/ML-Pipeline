#----------------------------------------------------------------------------
# LOADING NEW FEATURES

import os
import pandas as pd
import numpy as np

os.chdir("D:\BI\mltscn_combine_0.1_data")
print("Current working directory: ", os.getcwd())

def RecodeTarget(series):
    if series == 'Macro3':
        return 1
    elif series == 'Macro8':
        return 2
    else:
        return 0

dataset_top =  pd.read_csv('multi_full_731.csv', sep=";", decimal=",")
print("Number of rows: ", dataset_top.shape[0]) #4761
print("Number of features: ", dataset_top.shape[1]) #731

dop1 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Unique_mcc_num_by_periods_idresp.csv", sep=";", decimal=",")
print("Number of rows: ", dop1.shape[0]) #7418
print("Number of features: ", dop1.shape[1]) #13

dop2 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Trans_num_by_periods_idresp.csv", sep=";", decimal=",")
print("Number of rows: ", dop2.shape[0]) #7418
print("Number of features: ", dop2.shape[1]) #13

dop3 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Mean_num_days_between_trx_by_period.csv", sep=";", decimal=",")
print("Number of rows: ", dop3.shape[0]) #7322
print("Number of features: ", dop3.shape[1]) #13

dop3 = dop3.rename(columns={'january2017': 'num_day_btw_trs_january2017',
			'october2017': 'num_day_btw_trs_october2017',
			'november2017': 'num_day_btw_trs_november2017',
			'december2017': 'num_day_btw_trs_december2017',
       		'february2017': 'num_day_btw_trs_february2017',
			'march2017': 'num_day_btw_trs_march2017',
			'april2017': 'num_day_btw_trs_april2017',
			'may2017': 'num_day_btw_trs_may2017',
			'june2017': 'num_day_btw_trs_june2017',
       		'jule2017': 'num_day_btw_trs_jule2017',
			'august2017': 'num_day_btw_trs_august2017',
			'september2017': 'num_day_btw_trs_september2017'})

dop4 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Mean_num_days_between_trx_by_client.csv", sep=";", decimal=",")
print("Number of rows: ", dop4.shape[0]) #7322
print("Number of features: ", dop4.shape[1]) #2

dop5 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Diff_by_client.csv", sep=";", decimal=",")
print("Number of rows: ", dop5.shape[0]) #7418
print("Number of features: ", dop5.shape[1]) #4

dop6 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Days_diff_first_last_trans_by_period.csv", sep=";", decimal=",")
print("Number of rows: ", dop6.shape[0]) #7418
print("Number of features: ", dop6.shape[1]) #13

dop6 = dop6.rename(columns={'january2017': 'diff_days_lf_january2017',
			'october2017': 'diff_days_lf_october2017',
			'november2017': 'diff_days_lf_november2017',
			'december2017': 'diff_days_lf_december2017',
       			'february2017': 'diff_days_lf_february2017',
			'march2017': 'diff_days_lf_march2017',
			'april2017': 'diff_days_lf_april2017',
			'may2017': 'diff_days_lf_may2017',
			'june2017': 'diff_days_lf_june2017',
       			'jule2017': 'diff_days_lf_jule2017',
			'august2017': 'diff_days_lf_august2017',
			'september2017': 'diff_days_lf_september2017'})

dop7 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Days_diff_first_last_trans_by_client.csv", sep=";", decimal=",")
print("Number of rows: ", dop7.shape[0]) #7418
print("Number of features: ", dop7.shape[1]) #2

dop8 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Amount_diff_first_last_trans_by_period.csv", sep=";", decimal=",")
print("Number of rows: ", dop8.shape[0]) #7418
print("Number of features: ", dop8.shape[1]) #13

dop9 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Amount_diff_first_last_trans_by_client.csv", sep=";", decimal=",")
print("Number of rows: ", dop9.shape[0]) #7418
print("Number of features: ", dop9.shape[1]) #2

dop10 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Amount_by_client_sum_mean_std_min_max_p2p_idresp.csv", sep=";", decimal=",")
print("Number of rows: ", dop10.shape[0]) #5038
print("Number of features: ", dop10.shape[1]) #73

dop11 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Amount_by_client_sum_mean_std_min_max_idresp.csv", sep=";", decimal=",")
print("Number of rows: ", dop11.shape[0]) #7418
print("Number of features: ", dop11.shape[1]) #73

dop12 = pd.read_csv("D:/BI/mltscn_combine_0.1_data/dop_feat/Features/Amount_by_client_sum_mean_std_min_max_cash_idresp.csv", sep=";", decimal=",")
print("Number of rows: ", dop12.shape[0]) #6024
print("Number of features: ", dop12.shape[1]) #73

#----------------------------------------------------------------------------
# DATA CONTROL

# For trees

'''dop1 = dop1.replace(' ', -999)
dop2 = dop2.replace(' ', -999)
dop3 = dop3.replace(' ', -999)
dop4 = dop4.replace(' ', -999)
dop5 = dop5.replace(' ', -999)
dop6 = dop6.replace(' ', -999)
dop7 = dop7.replace(' ', -999)
dop8 = dop8.replace(' ', -999)
dop10 = dop10.replace(' ', -999)
dop11 = dop11.replace(' ', -999)
dop12 = dop12.replace(' ', -999)'''

# For Regression

dop1 = dop1.replace(' ', np.nan)
dop2 = dop2.replace(' ', np.nan)
dop3 = dop3.replace(' ', np.nan)
dop4 = dop4.replace(' ', np.nan)
dop5 = dop5.replace(' ', np.nan)
dop6 = dop6.replace(' ', np.nan)
dop7 = dop7.replace(' ', np.nan)
dop8 = dop8.replace(' ', np.nan)
dop10 = dop10.replace(' ', np.nan)
dop11 = dop11.replace(' ', np.nan)
dop12 = dop12.replace(' ', np.nan)

#----------------------------------------------------------------------------
# JOIN DATASETS

dataset = pd.merge(dataset_top, dop1,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop2,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop3,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop4,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop5,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop6,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop7,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop8,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop9,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop10,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop11,  how='left', on='id_resp')
dataset = pd.merge(dataset, dop12,  how='left', on='id_resp')

dop_soc = pd.read_csv("D:\BI\mltscn_combine_0.1_data\dop_feat\Features\multi_full_top_160518.csv", sep=";", decimal=",")

dop_soc_freq = dop_soc.loc[:, 'sport_n':'cash_amount_max_20']
dop_soc_freq['id_resp'] = dop_soc.id_resp

dataset = pd.merge(dataset, dop_soc_freq, how='left', on='id_resp')

# for trees
#dataset = dataset.replace(np.nan, -999)

del(dop1, dop2, dop3, dop4, dop5, dop6, dop7, dop8, dop9, dop10, dop11, dop12)

#----------------------------------------------------------------------------
# PREPROCESSING FOR LOGISTIC REGRESSION

colnames = pd.DataFrame(dataset.columns)

# Check and notice variables with nan

nadec = []
nadec_ind = []

for i in range(0, dataset.shape[1]):
	if dataset.iloc[:, i].isnull().values.any() == True:
		print("NA detected! Columns: ", dataset.iloc[:, i].name)
		nadec.append(dataset.iloc[:, i].name)
		nadec_ind.append(i)
	else:
		print('Ok')

# Notice names of future variables

listOfNames = []
for i in range(0, nadec_ind.__len__()):
	tmp = dataset.iloc[:, nadec_ind[i]].name + '_nan'
	listOfNames.append(tmp)

# Create label to nan for linear model

def newvargen(series):
	if pd.isnull(series) == True: 
		return 1
	else:
		return 0

# Applying "newvargen" to DataFrame
# Be careful! Slowly...

for i in range(0, nadec.__len__()):
	for j in range(0, listOfNames.__len__()):
		tmp = nadec[i]
		tmp0 = listOfNames[j]
		dataset[tmp0] = dataset[tmp].apply(newvargen)
		
#----------------------------------------------------------------------------
# Additional preprocessing

dataset['target38'] = dataset.target.apply(RecodeTarget)
group = pd.DataFrame(dataset.groupby('target38').count())
print(dataset.groupby('target38').groups.keys())
print("Nulls in dataset: ", dataset.isnull().values.any())
dataset=dataset.drop(columns=["target"])

# SAVE DATASET FOR FOREST

#dataset.to_csv("dataset_all_features_for_forest.csv")

# SAVE DATASET FOR REGRESSION

dataset.to_csv("dataset_all_features_for_regression.csv")