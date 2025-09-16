#!/usr/bin/env python
# coding: utf-8

# In[154]:


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


# In[156]:


df02 = df02.drop(columns = ['GP', 'Pos', 'TK', 'GV', '-9999'])
df02.head()


# In[157]:


df01['Player'] = df01['Player'].apply(unidecode).str.lower().str.strip()
df02 ['Player'] = df02['Player'].apply(unidecode).str.lower().str.strip()
df03 = pd.merge (df01, df02, how = 'left', on='Player')


# In[158]:


df03 = df03.drop(df03[df03['GP'] <= 41].index).reset_index(drop = True)

df03 = df03.replace({'RW': 'F', 'LW': 'F', 'C': 'F'})

df03 = df03.drop (df03[df03['Pos'] == 'G'].index).reset_index(drop = True)

df03 = df03.replace({'--': 0})

df03 = df03.fillna(0)


# In[159]:


df03.head()


# In[160]:


df_final = df03.copy()
df_final.head()
df_final = df_final.drop(columns = ['TOI/60'])


# In[161]:


df_final ['TOI(EV)'] = pd.to_timedelta('00:' + df_final['TOI(EV)'].astype(str)).dt.total_seconds()/60


# In[162]:


df_final = df_final.fillna(0)
df_final = df_final.replace({'--': 0})
df_final = df_final.drop(columns = ['Age', 'Pos'])


# In[163]:


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


# In[165]:


df.head()
df_time = df[['Player', 'TOI/GP']]


# In[166]:


df_time.head()


# In[167]:


df_time ['TOI/GP'] = pd.to_timedelta('00:' + df_time['TOI/GP'].astype(str)).dt.total_seconds()/60


# In[168]:


df_time['Player'] = df_time['Player'].apply(unidecode).str.lower().str.strip()


# In[169]:


nhl_df = pd.merge(df_final, df_time, how = 'left', on = 'Player')


# In[170]:


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



# In[171]:


nhl_df = nhl_df.drop (columns = 'SAtt.')
nhl_df.head()


# In[172]:


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


# In[173]:


nhl_df = nhl_df.drop(columns = ['GP', 'ATOI', 'TOI(EV)', 'TOI/GP'])
nhl_df.head()


# In[174]:


name_list = nhl_df['Player']
finalNHL_df = nhl_df.drop(columns = 'Player')
finalNHL_df.head()


# In[175]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

finalNHL_df = scale.fit_transform(finalNHL_df)


# In[176]:


from sklearn.cluster import KMeans

kmeans_nhl = KMeans(n_clusters = 12, random_state = 42)

cluster = kmeans_nhl.fit(finalNHL_df)


# In[177]:


print (kmeans_nhl.labels_)


# In[178]:


nhl_df['Clusters'] = kmeans_nhl.labels_


# In[179]:


nhl_df


# In[180]:


nhl_df


# In[181]:


for cluster_id in sorted(nhl_df["Clusters"].unique()):
    print(f"\nCluster {cluster_id}:")
    players_in_cluster = nhl_df[nhl_df["Clusters"] == cluster_id]["Player"].tolist()
    print(players_in_cluster)


# In[182]:


cluster_centroid = kmeans_nhl.cluster_centers_

centroid_df = pd.DataFrame(cluster_centroid, columns = nhl_df.drop(columns = ['Player', 'Clusters']).columns)



# In[183]:


centroid_df


# In[210]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(finalNHL_df)

nhl_df["PCA1"] = reduced[:,0]
nhl_df["PCA2"] = reduced[:,1]

# Plot
plt.figure(figsize=(10,7))
for cluster_id in nhl_df["Clusters"].unique():
    cluster_data = nhl_df[nhl_df["Clusters"] == cluster_id]
    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Clusters {cluster_id}", alpha=0.6)

plt.legend()
plt.title("Player Archetypes (KMeans + PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




