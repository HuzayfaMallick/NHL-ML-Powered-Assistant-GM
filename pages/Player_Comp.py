#!/usr/bin/env python
# coding: utf-8

# Final Model - Restructured

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


df01 = dfs('NHL_Final-01.csv')
df02 = dfs('NHL_Final-02.csv')


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
df_final = df_final.drop(columns = ['Age', 'Pos'])


# In[10]:


df_final.head()


# In[ ]:


df1 = dfs("NHL_STATS01.xlsx")
df2 = dfs("NHL_STATS02.xlsx")
df3 = dfs("NHL_STATS03.xlsx")
df4 = dfs("NHL_STATS04.xlsx")
df5 = dfs("NHL_STATS05.xlsx")
df6 = dfs("NHL_STATS06.xlsx")
df7 = dfs("NHL_STATS07.xlsx")
df8 = dfs("NHL_STATS08.xlsx")
df9 = dfs("NHL_STATS09.xlsx")
df10 = dfs("NHL_STATS10.xlsx")

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


# In[21]:


nhl_df01 = nhl_df.copy()

names_nhl = nhl_df01['Player']

nhl_df01 = nhl_df01.drop (columns = ['Player'])


# In[ ]:


salaries_df = dfs('salaries.csv')

salaries_df = salaries_df.drop(columns = ['0', '1', '2'])

salaries_df['Player'] = salaries_df['Player'].apply(unidecode).str.lower().str.strip()

names_df = pd.DataFrame({'Player' : names_nhl})


# In[23]:


names_df = pd.merge(names_df, salaries_df, how = 'left', on = 'Player')


# In[24]:


names_df.loc[names_df['Player'] == 'mitch marner', 'Salary'] = 10.903
names_df.loc[names_df['Player'] == 'jj peterka', 'Salary'] = 0.855834
names_df.loc[names_df['Player'] == 'matthew coronato', 'Salary'] = 0.925 
names_df.loc[names_df['Player'] == 'william cuylle', 'Salary'] = 0.828333
names_df.loc[names_df['Player'] == 'zachary bolduc', 'Salary'] = 0.863334
names_df.loc[names_df['Player'] == 'michael anderson', 'Salary'] = 4.125
names_df.loc[names_df['Player'] == 'emil martinsen lilleberg', 'Salary'] = 0.87
names_df.loc[names_df['Player'] == 'chris tanev', 'Salary'] = 4.5
names_df.loc[names_df['Player'] == 'alexey toropchenko', 'Salary'] = 1.25
names_df.loc[names_df['Player'] == 'fedor svechkov', 'Salary'] = 0.925
names_df.loc[names_df['Player'] == 'joe veleno', 'Salary'] = 0.9
names_df.loc[names_df['Player'] == 'egor zamula', 'Salary'] = 1.7
names_df.loc[names_df['Player'] == 'j.j. moser', 'Salary'] = 3.375
names_df.loc[names_df['Player'] == 'artem zub', 'Salary'] = 4.6
names_df.loc[names_df['Player'] == 't.j. brodie', 'Salary'] = 0.775
names_df.loc[names_df['Player'] == 'mathew dumba', 'Salary'] = 3.75
names_df.loc[names_df['Player'] == 'marc del gaizo', 'Salary'] = 0.775
names_df.loc[names_df['Player'] == 'emil andrae', 'Salary'] = 0.903
names_df.loc[names_df['Player'] == 'jonathon merrill', 'Salary'] = 1.2
names_df.loc[names_df['Player'] == 'devin shore', 'Salary'] = 0.775
names_df.loc[names_df['Player'] == 'zack ostapchuk', 'Salary'] = 0.825
names_df.loc[names_df['Player'] == 'matty beniers', 'Salary'] = 7.142857
names_df.loc[names_df['Player'] == 'oliver wahlstrom', 'Salary'] = 1.0



# In[25]:


names_df.head()


# In[26]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

nhl_scaled = scale.fit_transform(nhl_df01)


# In[27]:


from sklearn.neighbors import NearestNeighbors

sim_players = NearestNeighbors(n_neighbors = 550).fit(nhl_scaled)


# In[28]:


sample = 'charlie coyle'
index_pos = names_df.index.get_loc(names_df[names_df['Player'] == sample].index[0])
print (index_pos)


# In[29]:


result_comp = sim_players.kneighbors([nhl_scaled[241]])


# In[30]:


print(names_df['Salary'].loc[185])



# In[31]:


print(result_comp)


# In[32]:


print (result_comp [1][0][1])


# In[33]:


def comparables(comp_name, salary_max):

    #retrives position of comparison player in original array
    position = names_df.index.get_loc(names_df[names_df['Player'] == comp_name].index[0])

    #creates an array to score the most comparable players to the requested comparison
    comp_array = sim_players.kneighbors([nhl_scaled[position]])

    #variable assignment
    index = 1
    player_count = 0
    final_dict = {
        "name": [],
        "salary": [], 
        "similar_score": []
        }
    

    #for loop 
    for i in range(595): 

        #stores position of comparable
        pos_comp = comp_array[1][0][index]

        if names_df['Salary'].loc[pos_comp] <= salary_max:

            final_dict["name"].append(names_df['Player'].loc[pos_comp])
            final_dict["salary"].append(names_df['Salary'].loc[pos_comp].round(4))
            final_dict["similar_score"].append(comp_array[0][0][index].round(4))

            player_count += 1
            
        index += 1

        if player_count == 3:
            return final_dict
    
    


# In[34]:


print (comparables ("brady tkachuk", 8))


# In[ ]:




