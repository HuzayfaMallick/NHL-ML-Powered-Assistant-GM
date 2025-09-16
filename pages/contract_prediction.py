#!/usr/bin/env python
# coding: utf-8

# In[247]:


import pandas as pd
import numpy as np
from unidecode import unidecode


# In[248]:


df01 = pd.read_csv('NHL_Final-01.csv')
df02 = pd.read_csv('NHL_Final-02.csv')


# In[249]:


df02 = df02.drop(columns = ['GP', 'Pos', 'TK', 'GV', '-9999'])
df02.head()


# In[250]:


df01['Player'] = df01['Player'].apply(unidecode).str.lower().str.strip()
df02 ['Player'] = df02['Player'].apply(unidecode).str.lower().str.strip()
df03 = pd.merge (df01, df02, how = 'left', on='Player')


# In[251]:


df03 = df03.drop(df03[df03['GP'] <= 41].index).reset_index(drop = True)

df03 = df03.replace({'RW': 'F', 'LW': 'F', 'C': 'F'})

df03 = df03.drop (df03[df03['Pos'] == 'G'].index).reset_index(drop = True)

df03 = df03.replace({'--': 0})

df03 = df03.fillna(0)


# In[252]:


df03.head()


# In[253]:


df_final = df03.copy()
df_final.head()
df_final = df_final.drop(columns = ['TOI/60'])


# In[254]:


df_final ['TOI(EV)'] = pd.to_timedelta('00:' + df_final['TOI(EV)'].astype(str)).dt.total_seconds()/60


# In[255]:


df_final = df_final.fillna(0)
df_final = df_final.replace({'--': 0})
df_final = df_final.drop(columns = ['Pos'])


# In[256]:


df_final.head()


# In[257]:


df1 = pd.read_excel("NHL_STATS01.xlsx")
df2 = pd.read_excel("NHL_STATS02.xlsx")
df3 = pd.read_excel("NHL_STATS03.xlsx")
df4 = pd.read_excel("NHL_STATS04.xlsx")
df5 = pd.read_excel("NHL_STATS05.xlsx")
df6 = pd.read_excel("NHL_STATS06.xlsx")
df7 = pd.read_excel("NHL_STATS07.xlsx")
df8 = pd.read_excel("NHL_STATS08.xlsx")
df9 = pd.read_excel("NHL_STATS09.xlsx")
df10 = pd.read_excel("NHL_STATS10.xlsx")

df = pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index = True)


# In[258]:


df.head()
df_time = df[['Player', 'TOI/GP']]


# In[259]:


df_time.head()


# In[260]:


df_time ['TOI/GP'] = pd.to_timedelta('00:' + df_time['TOI/GP'].astype(str)).dt.total_seconds()/60


# In[261]:


df_time['Player'] = df_time['Player'].apply(unidecode).str.lower().str.strip()


# In[262]:


nhl_df = pd.merge(df_final, df_time, how = 'left', on = 'Player')


# In[263]:


nhl_df.loc[nhl_df['Player'] == 'mitch marner', 'TOI/GP'] = 21.3167
nhl_df.loc[nhl_df['Player'] == 'jj peterka', 'TOI/GP'] = 18.1833
nhl_df.loc[nhl_df['Player'] == 'matthew coronato', 'TOI/GP'] = 17.5833
nhl_df.loc[nhl_df['Player'] == 'william cuylle', 'TOI/GP'] = 15.0833
nhl_df.loc[nhl_df['Player'] == 'michael anderson', 'TOI/GP'] = 22.6667
nhl_df.loc[nhl_df['Player'] == 'jacob middleton', 'TOI/GP'] = 21.8667
nhl_df.loc[nhl_df['Player'] == 'pat maroon', 'TOI/GP'] = 11.5333
nhl_df.loc[nhl_df['Player'] == 'emil martinsen lilleberg', 'TOI/GP'] = 15.2833
nhl_df.loc[nhl_df['Player'] == 'nicklaus perbix', 'TOI/GP'] = 14.6833
nhl_df.loc[nhl_df['Player'] == 'cameron york', 'TOI/GP'] = 20.7833
nhl_df.loc[nhl_df['Player'] == 'joe veleno', 'TOI/GP'] = 12.1167
nhl_df.loc[nhl_df['Player'] == 'william borgen', 'TOI/GP'] = 17.0833
nhl_df.loc[nhl_df['Player'] == 'zachary jones', 'TOI/GP'] = 17.25
nhl_df.loc[nhl_df['Player'] == 't.j. brodie', 'TOI/GP'] = 15.6333
nhl_df.loc[nhl_df['Player'] == 'jonathon merrill', 'TOI/GP'] = 14.10
nhl_df.loc[nhl_df['Player'] == 'pierre-olivier joseph', 'TOI/GP'] = 15.2667



# In[264]:


nhl_df = nhl_df.drop (columns = 'SAtt.')
nhl_df.head()


# In[265]:


per_60_TOT = ['G', 'A', 'PTS', 'PIM', 'GWG', 'SOG', 'TSA', 'FOW', 'FOL', 'BLK', 'HIT', 'TAKE', 'GIVE']

for item in per_60_TOT:
    nhl_df[item] = (nhl_df[item] / (nhl_df['TOI/GP'] * nhl_df['GP']) * 60).round(5)
    nhl_df = nhl_df.rename(columns = {item: item + '/60'})


per_60_EV = ['EVG', 'EV', 'CF', 'CA', 'FF', 'FA']

for item in per_60_EV:
    nhl_df[item] = (nhl_df[item] / (nhl_df['TOI(EV)'] * nhl_df['GP']) * 60).round(5)
    nhl_df = nhl_df.rename(columns = {item: item + '/60'})

per_game = ['PPG', 'SHG', 'PP', 'SH']

for item in per_game:
    nhl_df[item] = (nhl_df[item] / nhl_df['GP']).round(5)
    nhl_df = nhl_df.rename(columns = {item: item + '/GP'})


# In[266]:


nhl_df = nhl_df.drop(columns = ['GP', 'ATOI', 'TOI(EV)', 'TOI/GP'])
nhl_df.head()


# In[267]:


salaries_df = pd.read_csv('salaries.csv')
salaries_df = salaries_df.drop(columns = ['0', '1', '2'])
salaries_df['Player'] = salaries_df['Player'].apply(unidecode).str.lower().str.strip()

nhl_df = pd.merge(nhl_df, salaries_df, how = 'left', on = 'Player')


# In[268]:


nhl_df.loc[nhl_df['Player'] == 'mitch marner', 'Salary'] = 10.903
nhl_df.loc[nhl_df['Player'] == 'jj peterka', 'Salary'] = 0.855834
nhl_df.loc[nhl_df['Player'] == 'matthew coronato', 'Salary'] = 0.925 
nhl_df.loc[nhl_df['Player'] == 'william cuylle', 'Salary'] = 0.828333
nhl_df.loc[nhl_df['Player'] == 'zachary bolduc', 'Salary'] = 0.863334
nhl_df.loc[nhl_df['Player'] == 'michael anderson', 'Salary'] = 4.125
nhl_df.loc[nhl_df['Player'] == 'emil martinsen lilleberg', 'Salary'] = 0.87
nhl_df.loc[nhl_df['Player'] == 'chris tanev', 'Salary'] = 4.5
nhl_df.loc[nhl_df['Player'] == 'alexey toropchenko', 'Salary'] = 1.25
nhl_df.loc[nhl_df['Player'] == 'fedor svechkov', 'Salary'] = 0.925
nhl_df.loc[nhl_df['Player'] == 'joe veleno', 'Salary'] = 0.9
nhl_df.loc[nhl_df['Player'] == 'egor zamula', 'Salary'] = 1.7
nhl_df.loc[nhl_df['Player'] == 'j.j. moser', 'Salary'] = 3.375
nhl_df.loc[nhl_df['Player'] == 'artem zub', 'Salary'] = 4.6
nhl_df.loc[nhl_df['Player'] == 't.j. brodie', 'Salary'] = 0.775
nhl_df.loc[nhl_df['Player'] == 'mathew dumba', 'Salary'] = 3.75
nhl_df.loc[nhl_df['Player'] == 'marc del gaizo', 'Salary'] = 0.775
nhl_df.loc[nhl_df['Player'] == 'emil andrae', 'Salary'] = 0.903
nhl_df.loc[nhl_df['Player'] == 'jonathon merrill', 'Salary'] = 1.2
nhl_df.loc[nhl_df['Player'] == 'devin shore', 'Salary'] = 0.775
nhl_df.loc[nhl_df['Player'] == 'zack ostapchuk', 'Salary'] = 0.825
nhl_df.loc[nhl_df['Player'] == 'matty beniers', 'Salary'] = 7.142857
nhl_df.loc[nhl_df['Player'] == 'oliver wahlstrom', 'Salary'] = 1.0


# In[269]:


nhl_df.head()


# In[270]:


names_list = nhl_df['Player']

FinalNhl_df = nhl_df.drop(columns={'Player'})

FinalNhl_df = FinalNhl_df.rename(columns={FinalNhl_df.columns[26]: 'CF%_rel'})

from sklearn.model_selection import train_test_split

X = FinalNhl_df.drop(columns = {'Salary'})
y = nhl_df['Salary']


# In[271]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42) 


# In[272]:


from sklearn.preprocessing import StandardScaler

scale_factor = StandardScaler()

X_train = scale_factor.fit_transform(X_train)
X_test = scale_factor.transform(X_test)


# In[273]:


from sklearn.ensemble import RandomForestRegressor

predictor = RandomForestRegressor(n_estimators=500, random_state = 42)

predictor.fit(X_train, y_train)


# In[274]:


salary_prediction = predictor.predict(X_test)


# In[275]:


print (salary_prediction)


# In[276]:


import shap

explainer = shap.TreeExplainer(predictor)

shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type = 'bar')


# In[277]:


print (FinalNhl_df.columns)


# In[278]:


feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

importance_df["perc_importance"] = importance_df["importance"] / importance_df["importance"].sum()
importance_df["cumulative_importance"] = importance_df["perc_importance"].cumsum()

print(importance_df.head(15))


# In[279]:


nhl_guess = pd.DataFrame([(nhl_df.drop(columns = ['Player'])).mean()])
nhl_guess = nhl_guess.drop(columns= 'Salary')
pd.set_option("display.max_columns", None)
avg_reset = nhl_guess.copy()
avg_reset = avg_reset.rename(columns={avg_reset.columns[26]: 'CF%_rel'})
nhl_guess.head()


# In[280]:


print(FinalNhl_df.columns[11])
print(FinalNhl_df.columns[0])
print(FinalNhl_df.columns[31])
print(FinalNhl_df.columns[3])
print(FinalNhl_df.columns[27])
print(FinalNhl_df.columns[7])
print(FinalNhl_df.columns[19])
print(FinalNhl_df.columns[32])
print(FinalNhl_df.columns[36])
print(FinalNhl_df.columns[22])
print(FinalNhl_df.columns[23])
print(FinalNhl_df.columns[4])
print(FinalNhl_df.columns[26])
print(FinalNhl_df.columns[28])
print(FinalNhl_df.columns[15])


# In[281]:


#contract_prediction_avg = predictor.predict(nhl_guess)

#print(contract_prediction_avg)


# In[282]:


nhl_guess = nhl_guess.rename(columns={nhl_guess.columns[26]: 'CF%_rel'})


# In[283]:


def contract_predictor (pp_gp, age, oiSH, pts_60, ff_60, ppg_gp, blk_60, oiSV, e_plus, give_60, cf_60, plus_minus, cf_rel, FA_60, TSA_60):
    nhl_guess = X.sample(1).copy()
    items_1 = ['PP/GP', 'Age', 'oiSH%', 'PTS/60', 'FF/60', 'PPG/GP', 'BLK/60', 'oiSV%', 'E+/-', 'GIVE/60', 'CF/60', '+/-', 'CF%_rel', 'FA/60', 'TSA/60']
    items_2 = [pp_gp, age, oiSH, pts_60, ff_60, ppg_gp, blk_60, oiSV, e_plus, give_60, cf_60, plus_minus, cf_rel, FA_60, TSA_60]
    count = 0
    for i in items_1:
        nhl_guess[i] = items_2[count]
        count+=1

    nhl_guess = nhl_guess[X.columns]
    train_model = scale_factor.transform(nhl_guess)
    return predictor.predict(train_model)


# In[284]:


row = nhl_df.loc[89]
print (row)


# In[ ]:





# In[ ]:




