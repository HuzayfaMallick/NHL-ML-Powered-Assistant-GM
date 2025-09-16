#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from unidecode import unidecode


# In[ ]:


import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_folder = os.path.join(root_dir, 'csv_files')

data_files = [f for f in os.listdir(data_folder) if f.endswith(('.csv', '.xlsx'))]

dfs = {}
for file in data_files:
    file_path = os.path.join(data_folder, file)
    if file.endswith('.csv'):
        dfs[file] = pd.read_csv(file_path)
    elif file.endswith('.xlsx'):
        dfs[file] = pd.read_excel(file_path, engine='openpyxl')


# In[ ]:


df01 = dfs['NHL_Final-01.csv']
df02 = dfs['NHL_Final-02.csv']


# In[3]:


df02 = df02.drop(columns = ['GP', 'Pos', 'TK', 'GV', '-9999'])
df02.head()


# In[4]:


df01['Player'] = df01['Player'].apply(unidecode).str.lower().str.strip()
df02 ['Player'] = df02['Player'].apply(unidecode).str.lower().str.strip()
df03 = pd.merge (df01, df02, how = 'left', on='Player')


# In[5]:


df03 = df03.drop(df03[df03['GP'] <= 41].index).reset_index(drop = True)

df03 = df03.replace({'RW': 'F', 'LW': 'F', 'C': 'F'})

df03 = df03.drop (df03[df03['Pos'] == 'G'].index).reset_index(drop = True)

df03 = df03.replace({'--': 0})

df03 = df03.fillna(0)


# In[6]:


df03.head()


# In[7]:


df_final = df03.copy()
df_final.head()
df_final = df_final.drop(columns = ['TOI/60'])


# In[8]:


df_final ['TOI(EV)'] = pd.to_timedelta('00:' + df_final['TOI(EV)'].astype(str)).dt.total_seconds()/60


# In[9]:


df_final = df_final.fillna(0)
df_final = df_final.replace({'--': 0})
df_final = df_final.drop(columns = ['Pos'])


# In[10]:


df_final.head()


# In[ ]:


df1 = dfs["NHL_STATS01.xlsx"]
df2 = dfs["NHL_STATS02.xlsx"]
df3 = dfs["NHL_STATS03.xlsx"]
df4 = dfs["NHL_STATS04.xlsx"]
df5 = dfs["NHL_STATS05.xlsx"]
df6 = dfs["NHL_STATS06.xlsx"]
df7 = dfs["NHL_STATS07.xlsx"]
df8 = dfs["NHL_STATS08.xlsx"]
df9 = dfs["NHL_STATS09.xlsx"]
df10 = dfs["NHL_STATS10.xlsx"]

df = pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index = True)


# In[12]:


df.head()
df_time = df[['Player', 'TOI/GP']]


# In[13]:


df_time.head()


# In[14]:


df_time ['TOI/GP'] = pd.to_timedelta('00:' + df_time['TOI/GP'].astype(str)).dt.total_seconds()/60


# In[15]:


df_time['Player'] = df_time['Player'].apply(unidecode).str.lower().str.strip()


# In[16]:


nhl_df = pd.merge(df_final, df_time, how = 'left', on = 'Player')


# In[17]:


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



# In[18]:


nhl_df = nhl_df.drop (columns = 'SAtt.')
nhl_df.head()


# In[19]:


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


# In[20]:


nhl_df = nhl_df.drop(columns = ['GP', 'ATOI', 'TOI(EV)', 'TOI/GP'])
nhl_df.head()


# In[ ]:


salaries_df = dfs['salaries.csv']
salaries_df = salaries_df.drop(columns = ['0', '1', '2'])
salaries_df['Player'] = salaries_df['Player'].apply(unidecode).str.lower().str.strip()

nhl_df = pd.merge(nhl_df, salaries_df, how = 'left', on = 'Player')


# In[22]:


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


# In[23]:


nhl_df.head()


# In[24]:


names_list = nhl_df['Player']

FinalNhl_df = nhl_df.drop(columns={'Player'})

FinalNhl_df = FinalNhl_df.rename(columns={FinalNhl_df.columns[26]: 'CF%_rel'})

from sklearn.model_selection import train_test_split

X = FinalNhl_df.drop(columns = {'Salary'})
y = nhl_df['Salary']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42) 


# In[26]:


from sklearn.preprocessing import StandardScaler

scale_factor = StandardScaler()

X_train = scale_factor.fit_transform(X_train)
X_test = scale_factor.transform(X_test)


# In[27]:


from sklearn.ensemble import RandomForestRegressor

predictor = RandomForestRegressor(n_estimators=500, random_state = 42)

predictor.fit(X_train, y_train)


# In[28]:


salary_prediction = predictor.predict(X_test)


# In[29]:


print (salary_prediction)


# In[30]:


import shap

explainer = shap.TreeExplainer(predictor)

shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type = 'bar')


# In[31]:


print (FinalNhl_df.columns)


# In[32]:


feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

importance_df["perc_importance"] = importance_df["importance"] / importance_df["importance"].sum()
importance_df["cumulative_importance"] = importance_df["perc_importance"].cumsum()

print(importance_df.head(15))


# In[33]:


nhl_guess = pd.DataFrame([(nhl_df.drop(columns = ['Player'])).mean()])
nhl_guess = nhl_guess.drop(columns= 'Salary')
pd.set_option("display.max_columns", None)
avg_reset = nhl_guess.copy()
avg_reset = avg_reset.rename(columns={avg_reset.columns[26]: 'CF%_rel'})
nhl_guess.head()


# In[34]:


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


# In[35]:


#contract_prediction_avg = predictor.predict(nhl_guess)

#print(contract_prediction_avg)


# In[36]:


nhl_guess = nhl_guess.rename(columns={nhl_guess.columns[26]: 'CF%_rel'})


# In[37]:


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


# In[39]:


row = nhl_df.loc[4]
print (row)


# In[ ]:





# In[ ]:




