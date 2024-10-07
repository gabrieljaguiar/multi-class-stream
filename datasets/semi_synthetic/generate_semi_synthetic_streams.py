from imblearn.over_sampling import SMOTE
import pandas as pd


def resample(dataframe):
    sm_1 = SMOTE(sampling_strategy="all", k_neighbors=3, random_state=42)
    sm_2 = SMOTE(sampling_strategy="all", k_neighbors=5, random_state=45)
    sm_3 = SMOTE(sampling_strategy="all", k_neighbors=3, random_state=48)
    sm_4 = SMOTE(sampling_strategy="all", k_neighbors=5, random_state=52)
    sm_5 = SMOTE(sampling_strategy="all", k_neighbors=3, random_state=56)
    
    X, y  = dataframe.drop(["class"], axis=1), dataframe.loc[:, "class"]
    X_res_1, y_res_1 = sm_1.fit_resample(X,y)
    X_res_2, y_res_2 = sm_2.fit_resample(X,y)
    X_res_3, y_res_3 = sm_3.fit_resample(X,y)
    X_res_4, y_res_4 = sm_4.fit_resample(X,y)
    X_res_5, y_res_5 = sm_5.fit_resample(X,y)
    
    X_res_partial = pd.concat([X_res_1, X_res_2, X_res_3, X_res_4, X_res_5], axis=0)
    y_res_partial = pd.concat([y_res_1, y_res_2, y_res_3, y_res_4, y_res_5], axis=0)
    
    X_res_all, y_res_all = sm_5.fit_resample(X_res_partial, y_res_partial)
    X_res_all = pd.concat([X_res_all, X_res_partial], axis=0)
    y_res_all = pd.concat([y_res_all, y_res_partial], axis=0)

    
    concept_resamp = pd.concat([X_res_all, y_res_all], axis = 1)
    
    return pd.concat([dataframe, concept_resamp], axis = 0).sample(frac=1, random_state=42).reset_index(drop=True)

concepts_breaks = [14352, 19500, 33240, 38682, 39510]

df = pd.read_csv("./INSECTS abrupt_balanced.csv", header=None)

df.columns = ["f_{}".format(i) for i in range(len(df.columns) - 1)] + ["class"]

df["class"] = df["class"].astype('str')
df["class"] = df["class"].astype('category')
df["class"] = df["class"].cat.codes


concept_1 = df.iloc[0:concepts_breaks[0]]
concept_2 = df.iloc[concepts_breaks[0]:concepts_breaks[1]]
concept_3 = df.iloc[concepts_breaks[1]:concepts_breaks[2]]
concept_4 = df.iloc[concepts_breaks[2]:concepts_breaks[3]]
concept_5 = df.iloc[concepts_breaks[3]:concepts_breaks[4]]
concept_6 = df.iloc[concepts_breaks[4]:]

concept_1_semi_synth = resample(concept_1)

concept_2_semi_synth = resample(concept_2)
concept_3_semi_synth = resample(concept_3)
concept_4_semi_synth = resample(concept_4)
concept_5_semi_synth = resample(concept_5)
concept_6_semi_synth = resample(concept_6)

concept_1_semi_synth.to_csv("semi_synth_concept_1.csv", index=False)
concept_2_semi_synth.to_csv("semi_synth_concept_2.csv", index=False)
concept_3_semi_synth.to_csv("semi_synth_concept_3.csv", index=False)
concept_4_semi_synth.to_csv("semi_synth_concept_4.csv", index=False)
concept_5_semi_synth.to_csv("semi_synth_concept_5.csv", index=False)
concept_6_semi_synth.to_csv("semi_synth_concept_6.csv", index=False)