from numpy.random import seed
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from data_util import NormalizeData

import matplotlib.pyplot as plt

my_seed = 2022
seed(my_seed)
from tensorflow import random
random.set_seed(my_seed)
import pandas as pd
import numpy as np
from data_util import filter_data_scd, load_dataESC, load_data_ACC, correl

from lifelines import CoxPHFitter

from sklearn.metrics import roc_curve
from sklearn.base import clone

def esc_model(dtable, value4nan):
    escX = load_dataESC(dtable, value4nan=value4nan)
    # Citation: O'mahony, 2014
    esc_model = [0.15939858, -0.00294271, 0.0259082, 0.00446131, 0.4583082, 0.82639195, 0.71650361, -0.01799934]
    Pscd = np.zeros((escX.shape[0], 1))
    for pat in range(escX.shape[0]):
        Pscd[pat] = esc_model[0] * escX[pat, 0] + esc_model[1] * escX[pat, 0] ** 2 + \
                    esc_model[2] * escX[pat, 1] + esc_model[3] * escX[pat, 2] + \
                    esc_model[4] * escX[pat, 3] + esc_model[5] * escX[pat, 4] + \
                    esc_model[6] * escX[pat, 5] + esc_model[7] * escX[pat, 6]
    escpred_scd = 1 - 0.998 ** np.exp(Pscd)
    return np.squeeze(escpred_scd)

esc_cutoff = 0.06
maxN_rads = 3 # maximum number of principal Radiomics
fusClf_0 = LogisticRegression(random_state=0, n_jobs=-1, max_iter=500, verbose=False, class_weight='balanced')

outcome_varname = 'outcome_scd'
normalize_rad_method  = 'unit_max' #'unit_max' # 'none' 'unit_vector' unit_max  std_norm
# Read Clinical data
dtable = pd.read_csv('./clinical_data_scd.csv', index_col=0)
dtable.drop(dtable[dtable.mat_fn.str.contains('no_match')].index, axis=0, inplace=True)
# Read radiomics data
rad_dtable = pd.read_csv('./radiomics_features_dataset_fullstack_4PublicCode.csv',index_col=0)  # minor updated with the word 'index' tag first column
# rad_dtable.drop('mat_fn', axis=1, inplace=True)  # remove mat_fn column; as it already exists in clinical table
dumtable = np.asarray(rad_dtable)
radict = {'mean': 0, 'median': 1, 'min': 2, 'max': 3} # The csv file contains 4 volumns: mean|median|min|max radiomcis calcualted over all slices

# concatenate min and max radiomics (over all slices)
nf_sl = int(dumtable.shape[1] / 4) # Get number of features per patient (note you have four statistics for each patient)
ColNames = pd.read_csv('./radiomics_names_944.csv') # filecontaining nales of radiomic features: will be used for interprtation
ColNames = list(ColNames.iloc[:, 0])
aa = np.concatenate((dumtable[:, radict['max'] * nf_sl:(radict['max'] + 1) * nf_sl] , \
     dumtable[:, radict['min'] * nf_sl:(radict['min'] + 1) * nf_sl]), axis=1)
dumtable = aa
reptColNames = [ColNames[i]+'__min' for i in range(len(ColNames))] # rename column names for interpretation
ColNames = ColNames+reptColNames

dumtable_uncorr = pd.DataFrame(dumtable, index=rad_dtable.index,columns=ColNames)  # same index of clinical table
grouped_feature_ls = []
correlated_groups  = []
corr_features = correl(dumtable_uncorr, 0.90)
uncorr_rad = [x for x in dumtable_uncorr.columns if x not in corr_features] # Remove correlated predictors
rad_dtable = dumtable_uncorr.loc[:, uncorr_rad].copy()
dumtable   = np.asarray(rad_dtable)

idx = np.nonzero(np.all(dumtable == 0, axis=0))[0]  # remove features that are all zeros in all patients
rad_dtable.drop(columns=rad_dtable.columns[idx], axis=1, inplace=True)

noSegIdx = np.nonzero(np.all(np.asarray(rad_dtable) == 0, axis=1))[0]  # patients without segmentation has all-zero radiomics entries
num_radiomics_features = rad_dtable.shape[-1]
dtable = pd.concat([dtable, rad_dtable], axis=1)  # now you have both radiomics and clinical data in one table
dtable.drop(index=dtable.index[noSegIdx], axis=0, inplace=True)
startIDX_rad = dtable.shape[-1] - num_radiomics_features

####################################################################################
filtered_dtable, excluded_dtable = filter_data_scd(dtable, cohort='all',
                                                   contour_type='all', min_fu_dur=0)  # apply inclusion criterion to internal dataset

print(' Dataset Size =  ', filtered_dtable.shape[0])
print(' Dataset - Positive Cases =  ', np.sum(filtered_dtable[outcome_varname]))

L= maxN_rads + 1 # number of radiomics features from 0 to maxN_rads
M= 3  # 3 models: RAD; RAD+ESC; RAD+ACC
AUCmatrix = np.zeros((L,M))
cutoffMatrix = np.zeros((L,M))

value4nan = filtered_dtable.median(axis=0, skipna=True).fillna(0)

# Predict, Evaluate ESC MODEL
Y = filtered_dtable[outcome_varname]
escpred = esc_model(filtered_dtable, value4nan)
auc_roc = metrics.roc_auc_score(Y, escpred > esc_cutoff)
AUCmatrix[0,1] = auc_roc
fpr_esc, tpr_esc, th_arr = roc_curve(Y, escpred >= esc_cutoff)
print('ESC train data: AUC = ', auc_roc)
# Predict, Evaluate ACC MODEL
accX = load_data_ACC(filtered_dtable, value4nan=value4nan)
accpred = np.sum(accX, axis=1)
auc_roc = metrics.roc_auc_score(Y, accpred > 0)
AUCmatrix[0, 2] = auc_roc
fpr_acc, tpr_acc, th_arr = roc_curve(Y, accpred > 0)
print('ACC Model: AUC = ', auc_roc )

#################################################################################
X = np.asarray(filtered_dtable.iloc[:, startIDX_rad::])
rad_features = NormalizeData(X, feats_axis=1,norm_type=normalize_rad_method)  # 'none' 'unit_vector' unit_max  std_norm
orig_rad_features = rad_features
transformer = PCA(n_components= maxN_rads, random_state=my_seed, whiten=True)
rad_features = transformer.fit_transform(rad_features)

for model_n_rad in range(1,L): # this is the number of rads; will subtract 1 for index below
    ## Univariate COX Analysis--RAD Only
    plt.figure(30+model_n_rad)
    dum = np.concatenate((np.expand_dims(rad_features[:,model_n_rad-1], axis=1),
                          np.expand_dims(filtered_dtable['outcome_fu_duration'].to_numpy(), axis=1),
                          np.expand_dims(Y, axis=1)), axis=1)
    data = pd.DataFrame(dum, columns=['Rad{i}'.format(i=model_n_rad), 'time', 'scd'])
    cpf = CoxPHFitter().fit(data, duration_col="time", event_col="scd")
    cpf.print_summary()
    cpf.plot(hazard_ratios=True)

    ## maximum correlated Radiomic feature witht he PCA radiomics
    xx = rad_features[:, model_n_rad - 1].reshape(1, -1)
    d = [cosine_similarity(xx, orig_rad_features[:, rad_i].reshape(1, -1)) for rad_i in
         range(orig_rad_features.shape[-1])]
    dm = np.max(d)
    ddm = np.where(d >= 0.8 * dm)[0]
    ColNames = list(rad_dtable.columns)
    [print('PCA Radiomic', model_n_rad, ' corresponds to Radiomics: ', ColNames[ii]) for ii in ddm]

    ### SINGLE RADIOMICS MODELS
    print('#####################  SINGLE RADIOMICS MODELS  #####################')
    print('#####################################################################')
    # Radiomics-Only Model  using single principal rads
    fusClf = clone(fusClf_0)
    fusClf.fit(np.expand_dims(rad_features[:,model_n_rad-1], axis=1),Y) # measure power of a single RAD
    prd_rad = fusClf.predict_proba(np.expand_dims(rad_features[:,model_n_rad-1], axis=1))[:, 1]
    auc_roc = metrics.roc_auc_score(Y, prd_rad)
    print('--RADIOMICS-Only, single RAD, i= {i} RADS: AUC = {AUC}'.format(i=model_n_rad,AUC=auc_roc))

    # Radiomics-ESC Model using single principal rads
    fusClf = clone(fusClf_0)  # class_weight is implicitly applied: 'balanced'
    hybridX = np.concatenate((np.expand_dims(rad_features[:,model_n_rad-1], axis=1),
                                  np.expand_dims(escpred > esc_cutoff, 1)), 1)
    fusClf.fit(hybridX, Y)  # measure power of a single RAD
    prd_escrad = fusClf.predict_proba(hybridX)[:, 1]
    auc_roc = metrics.roc_auc_score(Y, prd_escrad)
    print('--RADIOMICS-ESC, single RAD, i= {i} RADS: AUC = {AUC}'.format(i=model_n_rad, AUC=auc_roc))

    # Radiomics-ACC Model using single principal rads
    fusClf = clone(fusClf_0)  # class_weight is implicitly applied: 'balanced'
    hybridX = np.concatenate((np.expand_dims(rad_features[:,model_n_rad-1], axis=1),
                                  np.expand_dims(accpred > 0, 1)), 1)
    fusClf.fit(hybridX, Y)  # measure power of a single RAD
    prd_enhaccrad = fusClf.predict_proba(hybridX)[:, 1]
    auc_roc = metrics.roc_auc_score(Y, prd_enhaccrad)
    print('--RADIOMICS-ACC, single RAD, i= {i} RADS: AUC = {AUC}'.format(i=model_n_rad, AUC=auc_roc))

    ### Multiple RADIOMICS MODELS
    print('####################  MULTIPLE RADIOMICS MODELS  ####################', 'N = ', str(model_n_rad))
    print('#####################################################################')
    # Radiomics-Only Model  using MULTIPLE principal rads
    fusClf = clone(fusClf_0) # class_weight is implecitly applied: 'balanced'
    fusClf.fit(rad_features[:,:model_n_rad],Y) # measure power of a single RAD
    prd_rad = fusClf.predict_proba(rad_features[:,:model_n_rad])[:, 1]
    auc_roc = metrics.roc_auc_score(Y, prd_rad)
    AUCmatrix[model_n_rad,0] = auc_roc
    fpr_rad, tpr_rad, th_arr = roc_curve(Y, prd_rad)
    print('--RADIOMICS-Only, N RAD= {i} RADS: AUC = {AUC}'.format(i=model_n_rad,AUC=auc_roc))

    # Radiomics-ESC Model using MULTIPLE principal rads
    fusClf = clone(fusClf_0)  # class_weight is implicitly applied: 'balanced'
    hybridX_trn = np.concatenate((rad_features[:,:model_n_rad],
                                  np.expand_dims(escpred > esc_cutoff, 1)), 1)
    fusClf.fit(hybridX, Y)  # measure power of a single RAD
    prd_escrad = fusClf.predict_proba(hybridX)[:, 1]
    auc_roc = metrics.roc_auc_score(Y, prd_escrad)
    AUCmatrix[model_n_rad,1] = auc_roc
    fpr_Resc, tpr_Resc, th_arr = roc_curve(Y, prd_escrad)
    print('--RADIOMICS-ESC, N RAD= {i} RADS: AUC = {AUC}'.format(i=model_n_rad, AUC=auc_roc))

    # Radiomics-ACC Model using MULTIPLE principal rads
    fusClf = clone(fusClf_0)  # class_weight is implicitly applied: 'balanced'
    hybridX = np.concatenate((rad_features[:,:model_n_rad],
                                  np.expand_dims(accpred >0, 1)), 1)
    fusClf.fit(hybridX, Y)  # measure power of a single RAD
    prd_accrad = fusClf.predict_proba(hybridX)[:, 1]
    auc_roc = metrics.roc_auc_score(Y, prd_accrad)
    AUCmatrix[model_n_rad,2] = auc_roc
    fpr_Racc, tpr_Racc, th_arr = roc_curve(Y, prd_accrad)
    print('--RADIOMICS-ACC, N RAD= {i} RADS: AUC = {AUC}'.format(i=model_n_rad, AUC=auc_roc))
###############################################################################################
################################  COX REGRESSION ANALYSIS  ####################################
###############################################################################################
from lifelines import CoxPHFitter
## COX Analysis--RAD Only
plt.figure(41)
dum = np.concatenate((rad_features[:,:model_n_rad],
                      np.expand_dims(filtered_dtable['outcome_fu_duration'].to_numpy(),axis=1),
                      np.expand_dims(Y,axis=1)), axis=1)
data1 = pd.DataFrame(dum,columns=['Rad1','Rad2','Rad3','time','scd'])
cpf = CoxPHFitter().fit(data1,duration_col="time",event_col="scd")
cpf.print_summary()
cpf.plot(hazard_ratios=True)
####################################
## COX analysis--RAD-ESC
plt.figure(42)
dum = np.concatenate((np.expand_dims(escpred>=esc_cutoff, axis=1),
                      np.expand_dims(filtered_dtable['outcome_fu_duration'].to_numpy(),axis=1),
                      np.expand_dims(Y,axis=1)), axis=1)
data = pd.DataFrame(dum,columns=['ESC','time','scd'])
cpf = CoxPHFitter().fit(data,duration_col="time",event_col="scd")
print('ESC')
cpf.print_summary()
dum = np.concatenate((rad_features[:,:model_n_rad],
                      np.expand_dims(escpred>=esc_cutoff, axis=1),
                      np.expand_dims(filtered_dtable['outcome_fu_duration'].to_numpy(),axis=1),
                      np.expand_dims(Y,axis=1)), axis=1)
data2 = pd.DataFrame(dum,columns=['Rad1','Rad2','Rad3','ESC','time','scd'])
cpf = CoxPHFitter().fit(data2,duration_col="time",event_col="scd")
print('RAD-ESC')
cpf.print_summary()
cpf.plot(hazard_ratios=True)
####################################
## Cox regression analysis--RAD-ACC/AHA
plt.figure(43)
dum = np.concatenate((np.expand_dims(accpred>=1, axis=1),
                      np.expand_dims(filtered_dtable['outcome_fu_duration'].to_numpy(),axis=1),
                      np.expand_dims(Y,axis=1)), axis=1)
data = pd.DataFrame(dum,columns=['ACC/AHA','time','scd'])
cpf = CoxPHFitter().fit(data,duration_col="time",event_col="scd")
print('ACC/AHA')
cpf.print_summary()
dum = np.concatenate((rad_features[:,:model_n_rad],
                      np.expand_dims(accpred>=1, axis=1),
                      np.expand_dims(filtered_dtable['outcome_fu_duration'].to_numpy(),axis=1),
                      np.expand_dims(Y,axis=1)), axis=1)
data3 = pd.DataFrame(dum,columns=['Rad1','Rad2','Rad3','ACC','time','scd'])
cpf = CoxPHFitter().fit(data3,duration_col="time",event_col="scd")
print('RAD-ACC/AHA')
cpf.print_summary()
cpf.plot(hazard_ratios=True)
####################################
##############################  VIOLIN PLOT @ 50% Threshold  ##################################
###############################################################################################
import seaborn as sns
sns.set_theme(style="whitegrid")
inner_type = 'box'
plt.figure(51)
ax = sns.violinplot(x="ESC", y="Rad1", hue="scd", data=data2, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Low Risk', 'High Risk'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])
plt.figure(52)
ax = sns.violinplot(x="ACC", y="Rad1", hue="scd", data=data3, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Low Risk', 'High Risk'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])

plt.figure(55)
ax = sns.violinplot(x="ESC", y="Rad2", hue="scd", data=data2, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Low Risk', 'High Risk'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])
plt.figure(57)
ax = sns.violinplot(x="ACC", y="Rad2", hue="scd", data=data3, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Low Risk', 'High Risk'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])

plt.figure(60)
ax = sns.violinplot(x="ESC", y="Rad3", hue="scd", data=data2, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Low Risk', 'High Risk'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])
plt.figure(62)
ax = sns.violinplot(x="ACC", y="Rad3", hue="scd", data=data3, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Low Risk', 'High Risk'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])

plt.figure(71)
dum = pd.melt(data2, value_vars=['Rad1','Rad2','Rad3'], id_vars='scd')
ax = sns.violinplot(x="variable", y="value", hue="scd", data=dum, palette="Set2", split=True, inner= inner_type)
ax.set_xticklabels(['Radiomic 1','Radiomic 2','Radiomic 3'])
ax.legend(handles=ax.legend_.legendHandles, labels=['SCD-', 'SCD+'])

from data_util import plotAUC
legend = False
plt.figure(101)
plotAUC(fpr_esc, tpr_esc, AUCmatrix[0, 1], legend_txt='ESC', width=2, init_plot=True, legend=legend)
plotAUC(fpr_rad, tpr_rad, AUCmatrix[3, 0], legend_txt='LGE Radiomics', width=2, legend=legend)
plotAUC(fpr_Resc, tpr_Resc, AUCmatrix[3, 1], legend_txt='LGE Radiomics+ESC', width=2, legend=legend)
plt.legend(loc="lower right")

plt.figure(102)
plotAUC(fpr_acc, tpr_acc, AUCmatrix[0, 2], legend_txt='ACC/AHA', width=2, init_plot=True, legend=legend)
plotAUC(fpr_rad, tpr_rad, AUCmatrix[3, 0], legend_txt='LGE Radiomics', width=2, legend=legend)
plotAUC(fpr_Racc, tpr_Racc, AUCmatrix[3, 2], legend_txt='LGE Radiomics + ACC/AHA', width=2, legend=legend)
plt.legend(loc="lower right")

plt.show()
