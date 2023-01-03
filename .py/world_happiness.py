


#  Features Explanation
# * Country : Ülke adı
# * Region  : Ülkenin bulunduğu bölge
# * Happiness Rank : Mutluluk puanına göre ülkenin sıralaması
# * Happiness Score: Mutluluk puanı
# * Economy (GDP per Capita) :  Kişi başına düşen GSYİH'in mutluluk puanına katkı oranı
# * Family : Ailenin mutluluk puanına katkı oranı
# * Health (Life Expectancy) : Sağlığın  mutluluk puanına katkı oranı
# * Freedom : Özğürlüğün mutluluk puanına katkı oranı
# * Trust : Yolsuzluk algısının mutluluk puanına katkı oranı
# * Generosity  : Cömertliğin mutluluk puanına katkı oranı




import shap
import sklearn
import geopandas
import numpy as np
import plotly as py
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from sklearn import metrics
from sklearn import ensemble
from sklearn import metrics 
from wordcloud import WordCloud
from sklearn import linear_model
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)


# # Data Exploration



import warnings
warnings.filterwarnings('ignore')




x = 2015
dff = []
while True:
    globals()[f'df{x}'] = pd.read_csv(f'/kaggle/input/world-happiness/{x}.csv')
    dff.append(globals()[f'df{x}'])
    x += 1
    if x == 2020:
        break



for i, df in enumerate(dff):
    print (f'201{i+5} dataset:')
    display (dff[i].head(3))


# # Data Cleaning




df2015.rename(columns = {'Economy (GDP per Capita)' : 'GDP',
                        'Health (Life Expectancy)' : 'Life',
                        'Trust (Government Corruption)' : 'Trust'}, inplace = True)





df2016.rename(columns = {'Economy (GDP per Capita)' : 'GDP',
                        'Health (Life Expectancy)' : 'Life',
                        'Trust (Government Corruption)' : 'Trust'}, inplace = True)



df2017.rename(columns = {'Happiness.Rank' : 'Happiness Rank',
                        'Happiness.Score' : 'Happiness Score',
                        'Economy..GDP.per.Capita.' : 'GDP',
                        'Health..Life.Expectancy.' : 'Life',
                        'Dystopia.Residual' : 'Dystopia Residual',
                        'Trust..Government.Corruption.' : 'Trust'}, inplace = True)




df2018.rename(columns = {'Overall rank' : 'Happiness Rank',
                        'Score' : 'Happiness Score',
                        'Country or region' : 'Country',
                        'Social support' : 'Family',
                        'Freedom to make life choices' : 'Freedom',
                        'GDP per capita' : 'GDP',
                        'Healthy life expectancy' : 'Life',
                        'Perceptions of corruption' : 'Trust'}, inplace = True)




df2019.rename(columns = {'Overall rank' : 'Happiness Rank',
                        'Score' : 'Happiness Score',
                        'Country or region' : 'Country',
                        'Social support' : 'Family',
                        'Freedom to make life choices' : 'Freedom',
                        'GDP per capita' : 'GDP',
                        'Healthy life expectancy' : 'Life',
                        'Perceptions of corruption' : 'Trust'}, inplace = True)




for i, df in enumerate(dff, 2015):
    df['Year'] = i




for df in dff:
    if not ('Region') in df:
        df['Region'] = None
        temp = df.set_index('Country').Region.fillna(df2015.set_index('Country').Region).reset_index()
        df.fillna(temp, inplace = True)




for i, df in enumerate(dff, 2015):
    print ('\n' f'df{i} dataset:' '\n', df.isnull().sum())




fuldf = pd.concat(dff)


# # Data Visualisation



df2015 = pd.read_csv("../input/world-happiness/2015.csv")
df2015.drop("Standard Error", axis=1,inplace=True)
df2015["Year"] = 2015
df2015.columns = ["Country","Region","Happiness Rank","Happiness Score","GDP","Family","Life","Freedom","Trust","Generosity","Dystopia Residual","Year"]

df2016 = pd.read_csv("../input/world-happiness/2016.csv")
df2016.drop(["Lower Confidence Interval", "Upper Confidence Interval"],axis=1,inplace=True)
df2016["Year"] = 2016
df2016.columns=["Country", "Region", "Happiness Rank", "Happiness Score", "GDP", "Family", "Life", "Freedom", "Trust", "Generosity", "Dystopia Residual", "Year"]

df2017 = pd.read_csv("../input/world-happiness/2017.csv")
df2017.drop(["Whisker.high","Whisker.low"],axis=1,inplace=True)
df2017["Year"]= 2017
df2017.columns=["Country", "Happiness Rank", "Happiness Score", "GDP", "Family", "Life", "Freedom","Generosity", "Trust", "Dystopia Residual", "Year"]


country_region = df2015[["Country","Region"]]
country_region = country_region.to_numpy()

def make_region(Country):
    for i in range(len(country_region)):
        if Country == country_region[i][0]:
            return country_region[i][1]
    
    return "no_region"

df2017["Region"] = df2017["Country"].apply(make_region)

df2017.loc[32,"Region"] = 'Eastern Asia'
df2017.loc[49,"Region"] = 'Latin America and Caribbean'
df2017.loc[70,"Region"] = 'Eastern Asia'
df2017.loc[92,"Region"] = 'Sub-Saharan Africa'
df2017.loc[110,"Region"] = 'Sub-Saharan Africa'
df2017.loc[146,"Region"] =  'Sub-Saharan Africa'

df2018 = pd.read_csv("../input/world-happiness/2018.csv")
df2018["Year"] =2018
df2018.columns = [ "Happiness Rank","Country", "Happiness Score", "GDP", "Family", "Life", "Freedom","Generosity", "Trust", "Year"]
df2018["Dystopia Residual"] = df2018["Happiness Score"] - df2018["GDP"] - df2018["Family"] - df2018["Life"] - df2018["Freedom"] - df2018["Generosity"] - df2018["Trust"]
df2018["Region"] = df2018["Country"].apply(make_region)
df2018.loc[37,"Region"] = 'Latin America and Caribbean'
df2018.loc[48,"Region"] = 'Latin America and Caribbean'
df2018.loc[57,"Region"] = 'Central and Eastern Europe'
df2018.loc[97,"Region"] = 'Sub-Saharan Africa'
df2018.loc[118,"Region"] = 'Sub-Saharan Africa'
df2018.loc[153,"Region"] = 'Sub-Saharan Africa'

df2019 =pd.read_csv("../input/world-happiness/2019.csv")
df2019["Year"] = 2019
df2019.columns =[ "Happiness Rank","Country", "Happiness Score", "GDP", "Family", "Life", "Freedom","Generosity", "Trust", "Year"]
df2019["Dystopia Residual"] = df2019["Happiness Score"] - df2019["GDP"] - df2019["Family"] - df2019["Life"] - df2019["Freedom"] - df2019["Generosity"] - df2019["Trust"]
df2019["Region"] = df2019["Country"].apply(make_region)
df2019.loc[38,"Region"] = 'Latin America and Caribbean'
df2019.loc[63,"Region"] = 'Central and Eastern Europe'
df2019.loc[83,"Region"] = 'Central and Eastern Europe'
df2019.loc[111,"Region"] = 'Sub-Saharan Africa'
df2019.loc[112,"Region"] = 'Sub-Saharan Africa'
df2019.loc[119,"Region"] = 'Sub-Saharan Africa'
df2019.loc[155,"Region"] = 'Sub-Saharan Africa'




df_tmp = fuldf.groupby(["Region","Year"])["Happiness Score"].mean()
df_tmp = pd.DataFrame(df_tmp).unstack()
df_tmp = df_tmp.reset_index()
df_tmp.columns=["Region","2015","2016","2017","2018","2019"]
df_tmp = df_tmp.melt("Region")
fig = px.bar(df_tmp.sort_values(by="value"), x="Region", y="value",animation_frame='variable',text="value",color="Region",
             height=600,title = 'Yıllara göre bölgelerin mutluluk oranının değişimi')
fig.update_traces(texttemplate='%{text:.2f}',textposition='auto')
fig.update_layout(transition = {'duration': 1000})
fig.show()




df_1 = pd.concat([df2015.melt(id_vars=['Country','Year','Happiness Score','Happiness Rank','Region']),
                 df2016.melt(id_vars=['Country','Year','Happiness Score','Happiness Rank','Region']),],ignore_index=True)
fig = px.bar(df_1[df_1["Happiness Rank"] <=10].sort_values(by="Happiness Score"), y="Country", x="value", color='variable',animation_frame='Year',
             height=700,title="2015-2016 Yılları Arası Mutluluk Değişimi",opacity=.2,text="value")
fig.update_traces(texttemplate='%{text:.2f}',textposition='auto')
fig.update_layout(transition = {'duration': 1000})
fig.show()




df_2 = pd.concat([df2016.melt(id_vars=['Country','Year','Happiness Score','Happiness Rank','Region']),
                 df2017.melt(id_vars=['Country','Year','Happiness Score','Happiness Rank','Region']),],ignore_index=True)
fig = px.bar(df_2[df_2["Happiness Rank"] <=10].sort_values(by="Happiness Score"), y="Country", x="value", color='variable',animation_frame='Year',
             height=700,title="2016-2017 Yılları Arası Mutluluk Değişimi",opacity=.2,text="value")
fig.update_traces(texttemplate='%{text:.2f}',textposition='auto')
fig.update_layout(transition = {'duration': 1000})
fig.show()





df1 = pd.read_csv('../input/world-happiness/2015.csv')



df2 = pd.read_csv('../input/world-happiness/2016.csv')




df3 = pd.read_csv('../input/world-happiness/2017.csv')





df4 = pd.read_csv('../input/world-happiness/2018.csv')



df5 = pd.read_csv('../input/world-happiness/2019.csv')



import plotly.express as px
happiest_countries = fuldf.groupby(['Country'], sort = False)['Happiness Score', 'Year', 'GDP'].max()
top10 = happiest_countries.sort_values('Happiness Score', ascending = False)[:15]
fig = px.scatter(top10,
                x = top10.index,
                y = 'Happiness Score',
                size = 'GDP',
                color = top10.index,
                template = 'xgridoff',
                animation_frame = 'Year',
                title = 'En Mutlu 15 Ülke')
fig.show()



df2015 = df1.iloc[:20,:]
df2016 = df2.iloc[:20,:]
df2017 = df3.iloc[:20,:]
df2018 = df4.iloc[:20,:]
df2019 = df5.iloc[:20,:]

import plotly.graph_objs as go
v1 =go.Scatter(
                    x = df2015['Country'],
                    y = df2015['Happiness Score'],
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'red'),
                    text= df2015.Country)

v2 =go.Scatter(
                    x = df2015['Country'],
                    y = df2016['Happiness Score'],
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'green'),
                    text= df2016.Country)

v3 =go.Scatter(
                    x = df2015['Country'],
                    y = df2017['Happiness.Score'],
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'blue'),
                    text= df2017.Country)


v4 =go.Scatter(
                    x = df2015['Country'],
                    y = df2018['Score'],
                    mode = "markers",
                    name = "2018",
                    marker = dict(color = 'black'),
                    text= df2017.Country)


v5 =go.Scatter(
                    x = df2015['Country'],
                    y = df2019['Score'],
                    mode = "markers",
                    name = "2019",
                    marker = dict(color = 'pink'),
                    text= df2017.Country)


data = [v1, v2, v3, v4, v5]
layout = dict(title = 'En iyi 20 ülke arasındaki mutluluk oranının yıllara göre değişimi',
              xaxis= dict(title= 'Country',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Happiness',ticklen= 5,zeroline= False),
              hovermode="x unified"
             )
fig = dict(data = data, layout = layout)
iplot(fig)




data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
         marker_line_width=1,
        locations = df1['Country'],
        locationmode = "country names",
        z = df1['Happiness Score'],
        text = df1['Country'],
        colorbar = {'title' : 'Happiness Score'},
        
      )
layout = dict(title = ' 2015 yılı Dünya Mutluluk Haritası',
              geo = dict(projection = {'type':'mercator'}, showocean = False, showlakes = True, showrivers = True, )
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)




data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
         marker_line_width=1,
        locations = df2['Country'],
        locationmode = "country names",
        z = df2['Happiness Score'],
        text = df2['Country'],
        colorbar = {'title' : 'Happiness Score'},
        
      )
layout = dict(title = ' 2016 yılı Dünya Mutluluk Haritası',
              geo = dict(projection = {'type':'mercator'}, showocean = False, showlakes = True, showrivers = True, )
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)



data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
         marker_line_width=1,
        locations = df3['Country'],
        locationmode = "country names",
        z = df3['Happiness.Score'],
        text = df3['Country'],
        colorbar = {'title' : 'Happiness Score'},
        
      )
layout = dict(title = ' 2017 yılı Dünya Mutluluk Haritası',
              geo = dict(projection = {'type':'mercator'}, showocean = False, showlakes = True, showrivers = True, )
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)




fuldf['Happiness Change'] = (df5['Score'] - df1['Happiness Score']) / df1['Happiness Score']
temp = fuldf[np.abs(fuldf['Happiness Change']) > 0.01]
temp = fuldf.sort_values('Happiness Change')
temp['Year'] = temp['Year'].astype(str)
fig = px.bar(temp,
             x = 'Happiness Change',
             y = 'Country',
             color = 'Year',
             orientation = 'h',
             height = 900,
             template = 'gridon',
             title = '2015-2017 yılları arasında mutluluk oranının değişimi')
fig.show()

 


fig = px.scatter(fuldf,
                x = 'GDP',
                y = 'Happiness Score',
                size = 'Freedom',
                color = 'Country',
                template = 'xgridoff',
                animation_frame = 'Year',
                title = '2015 - 2019 Yılları arasındaki GDP * Mutluluk Oranını ve kabarcık Özğürlüğün Mutluluk oranına katkıtısını gösterir') 
fig.show()


 


fig = px.scatter(fuldf,
                x = 'Life',
                y = 'Happiness Score',
                size = 'GDP',
                color = 'Country',
                template = 'xgridoff',
                animation_frame = 'Year',
                labels = {'Life': 'Life Expectancy'},
                title = '2015 - 2019 yılları arasında ki Sağlıgın * Mutluluk oranının ve kabarcıklar GDP mutluluk oranına katkısı ')
fig.show()





import plotly.express as px

df2015 = pd.read_csv('../input/world-happiness/2015.csv')
df = df2015

fig = px.sunburst(df, path=['Region', 'Country'], values='Happiness Score',
                  color='Happiness Score', hover_data=['Happiness Rank'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['Happiness Score'], weights=df['Happiness Score']))
fig.update_layout(hovermode="x unified")
fig.show()

 


fuldf.drop(labels=['Upper Confidence Interval'],axis=1,inplace=True)
fuldf.drop(labels=['Dystopia Residual'],axis=1,inplace=True)
fuldf.drop(labels=['Whisker.high'],axis=1,inplace=True)
fuldf.drop(labels=['Whisker.low'],axis=1,inplace=True)
fuldf.drop(labels=['Standard Error'],axis=1,inplace=True)

 

fuldf.drop(labels=['Lower Confidence Interval'],axis=1,inplace=True)

 

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(fuldf.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)
plt.show()

 


fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(12,8))

sns.barplot(x='GDP',y='Country',data=fuldf.nlargest(10,'GDP'),ax=axes[0,0],palette="Blues_d")

sns.barplot(x='Family' ,y='Country',data=fuldf.nlargest(10,'Family'),ax=axes[0,1],palette="YlGn")

sns.barplot(x='Life' ,y='Country',data=fuldf.nlargest(10,'Life'),ax=axes[1,0],palette='OrRd')

sns.barplot(x='Freedom' ,y='Country',data=fuldf.nlargest(10,'Freedom'),ax=axes[1,1],palette='YlOrBr')

 


fig, axes = plt.subplots(nrows=1, ncols=2,constrained_layout=True,figsize=(10,4))

sns.barplot(x='Generosity' ,y='Country',data=fuldf.nlargest(10,'Generosity'),ax=axes[0],palette='Spectral')
sns.barplot(x='Trust' ,y='Country',data=fuldf.nlargest(10,'Trust'),ax=axes[1],palette='RdYlGn')


# Data içerinde bulunan öznitelikleri kategorik olarak sıralamak ve ayrım yapmak.
 

fuldf.tail()

 


fuldf = fuldf.rename(columns = {'Happiness Score': 'Score' })
fuldf['Trust'].fillna(value=fuldf['Trust'].mean(),inplace=True)
fuldf['Happiness Change'].fillna(value=fuldf['Happiness Change'].mean(),inplace=True)

 


fulldf=pd.concat(dff)
fulldf = fulldf.rename(columns = {'Happiness Score': 'Score' })
fulldf=fulldf[['Country','Score','Region','GDP']]
fulldf

 

fulldf=fulldf.groupby(['Country']).mean().reset_index()
fulldf

 


print('max:',fulldf['Score'].max())
print('min:',fulldf['Score'].min())
abc=fulldf['Score'].max()-fulldf['Score'].min()
scr=round(abc/3,3)
print('aradaki fark:',(scr))

 


düsük=fulldf['Score'].min()+scr
orta=düsük+scr

print('düsük scr in üst sınırı',düsük)
print('orta scr in üst sınırı',orta)
print('yüksek scr in üst sınırı','max:',fulldf['Score'].max())

 


sıra=[]
for i in fulldf.Score:
    if(i>0 and i<düsük):
        sıra.append('Düşük')
        
        
    elif(i>düsük and i<orta):
         sıra.append('Orta')
    else:
         sıra.append('Yüksek')

fulldf['Category']=sıra

 


color = (fulldf.Category == 'Yüksek' ).map({True: 'background-color: red ',False:'background-color: yellow',True: 'background-color: limegreen'})
fulldf.reset_index(drop=True).style.apply(lambda s: color)

 


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

 

fulldf = world.merge(fulldf, how="left", left_on=['name'], right_on=['Country'])

 

fig, ax = plt.subplots(figsize  = (12, 8))
ax.set_title("5 Yıllık Ortalama Mutluluk Kategorileri", fontsize=20)
fulldf.plot(column='Category',ax=ax,legend=True,cmap='prism')

 


df_tr = df_tr = fuldf["Country"]=="Turkey"
fuldf[df_tr]

 


df_turkey = fuldf[fuldf["Country"] =="Turkey"]
fig = px.line(df_turkey, x='Year', y='Happiness Rank', color='Country')
fig.update_layout(
    yaxis = dict(autorange="reversed")
)
fig.show()

 

df_fitre = (fuldf['Country']=="Turkey")| (fuldf['Country']=="Greece") | (fuldf["Country"]=="Armenia") | (fuldf["Country"]=="Syria")
fuldf[df_fitre]

 


countries = [i for i in fuldf[df_fitre]["Country"]]

features = ["GDP","Family","Life","Freedom","Score"]

colors = ["Blue","Cyan","Red","Green","Brown","Pink"]

features_colors = list(zip(features,colors))

def barplot_creator(country_list,feature_color_list,fuldf):
    for f,c in feature_color_list:
        _,ax = plt.subplots(figsize = (6,4))
        ax.bar(country_list,fuldf[f],color=c,label=f)
        plt.legend(loc = "upper right")
        plt.show()
        

barplot_creator(countries,features_colors,fuldf[df_fitre])

 


df_hapy = fuldf.pivot_table(index=["Country","Year"],values="Score")
df_hapy = df_hapy.unstack()
df_hapy = df_hapy.reset_index()
df_hapy.columns = ["Country","2015","2016","2017","2018","2019"]
df_hapy  = df_hapy.dropna()
df_hapy  = df_hapy.reset_index(drop=True)
for i in range(len(df_hapy)):
    df_hapy.loc[i,"min"] = min(df_hapy.loc[i,"2015"],df_hapy.loc[i,"2016"],df_hapy.loc[i,"2017"],df_hapy.loc[i,"2018"],df_hapy.loc[i,"2019"])

for i in range(len(df_hapy)):
    df_hapy.loc[i,"max"] = max(df_hapy.loc[i,"2015"],df_hapy.loc[i,"2016"],df_hapy.loc[i,"2017"],df_hapy.loc[i,"2018"],df_hapy.loc[i,"2019"])

for i in range(len(df_hapy)):
    df_hapy.loc[i,"difference"] = df_hapy.loc[i,"max"]-df_hapy.loc[i,"min"]

 


df_hapy.sort_values(by="difference",ascending=False).head(2)

 


df_venezuela = fuldf[fuldf["Country"] =="Venezuela"]

fig = px.bar(df_venezuela,x="Year",y="Score",color="Score",text="Score",title="Venezuela'nın Mutluluk Azalması")
fig.update_traces(texttemplate='%{text:.2f}',textposition='auto')
fig.show()

 


df_Benin = fuldf[fuldf["Country"]=="Benin"]
fig = px.bar(df_Benin,x="Year",y="Score",color="Score",text="Score",title="Benin'in mutluluk artışı")
fig.update_traces(texttemplate='%{text:.2f}',textposition='auto')
fig.show()


# # Machine Learning
 

X = fuldf.drop(['Score', 'Happiness Rank', 'Country','Region'],axis=1)
y = fuldf['Score']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Linear Model
 

lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)

sonuc_lm = pd.DataFrame({
    'Gerçek':y_test,
    'Tahmin':y_pred
})
sonuc_lm['Fark'] = y_test - y_pred
sonuc_lm.head()

 


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(f"R^2 of train set{lm.score(X_train, y_train)}")
print(f"R^2 of test set{lm.score(X_test, y_test)}")
sns.regplot(x='Gerçek',y='Tahmin',data=sonuc_lm)


# Random Forest Regressor
 


rf = ensemble.RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

sonuc_rf = pd.DataFrame({
    'Gerçek':y_test,
    'Tahmin':y_pred
})
sonuc_rf['Fark'] = y_test - y_pred
sonuc_rf.head()

 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(f"R^2 of train set{rf.score(X_train, y_train)}")
print(f"R^2 of test set{rf.score(X_test, y_test)}")

sns.regplot(x='Gerçek',y='Tahmin',data=sonuc_rf)

 


parameters = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1,2,3,4],
}
regr = ensemble.RandomForestRegressor(random_state=0)

clf = GridSearchCV(regr, parameters)
clf.fit(X_train, y_train)

 


y_pred_train = clf.predict(X_train)
metrics.mean_squared_error(y_train, y_pred_train)


 


y_pred = clf.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)


# # XGBoot
 


regressor=xgb.XGBRegressor(eval_metric='rmsle')
param_grid = {"max_depth":    [4, 5],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}
search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)

print("En İyi Hiperparametreler ",search.best_params_)

 


regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],)

regressor.fit(X_train, y_train)

 


 
 

sonuc_xg = pd.DataFrame({
    'Gerçek':y_test,
    'Tahmin':y_pred
})
sonuc_xg['Fark'] = y_test - y_pred
sonuc_xg.head()

 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(f"R^2 of train set{regressor.score(X_train, y_train)}")
print(f"R^2 of test set{regressor.score(X_test, y_test)}")

sns.regplot(x='Gerçek',y='Tahmin',data=sonuc_xg)


# # MODEL EXPLAINABILITY
 


regressor = ensemble.RandomForestRegressor()
regressor.fit(X_train, y_train);

 



explainer = shap.TreeExplainer(regressor)

shap_values = explainer.shap_values(X_train)

 

features_1 = X.columns

 


shap.summary_plot(shap_values, X_train, feature_names=X.columns)

 


shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_names = list(features_1))

 

i = 18
shap.force_plot(explainer.expected_value, shap_values[i], X_test[i], feature_names = features_1,matplotlib=True)
 


shap.summary_plot(shap_values,X, plot_type="bar")

 


Explainer = shap.TreeExplainer(rf)


 


shap.initjs()

 


shap_interaction_values = shap.TreeExplainer(rf).shap_interaction_values(X.iloc[:2000,:])

 


shap.summary_plot(shap_interaction_values, X.iloc[:2000,:])

 

explainer1 = shap.TreeExplainer(rf)
shap_values1 = explainer1.shap_values(X)

 


shap.force_plot(explainer1.expected_value, shap_values1[:1000,:], X.iloc[:1000,:])


# ## Clustering
 


labels = KMeans(2, random_state=0).fit_predict(X)
fuldf['labels'] = labels
tsne_data  = fuldf.drop(['Country', 'Region', 'Happiness Rank', 'Year', 'Happiness Change'],axis=1)
tsne_data["labels"] = tsne_data["labels"].astype(str)

 

wcss=[]
for i in range(1,15):
    km=KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1,15),wcss,"-o")
plt.grid(True)
plt.xlabel("k values")
plt.ylabel("wcss values")
plt.show()

 


X_1  = fuldf.drop(['Score', 'Happiness Rank', 'Country', 'Region','Trust','Freedom','Family','Generosity','Happiness Change','Year','labels'],axis=1)

 


df_2=fuldf.copy()
km2=KMeans(n_clusters=2)
clusters=km2.fit_predict(X_1)
df_2["clusters"]=clusters




df_1=fuldf.copy()
km1=KMeans(n_clusters=3)
clusters=km1.fit_predict(X_1)
df_1["clusters"]=clusters

 


plt.subplot(1,2,1)
plt.scatter(df_2["GDP"][df_2.clusters==0],
            df_2["Life"][df_2.clusters==0],color="b")

plt.scatter(df_2["GDP"][df_2.clusters==1],
            df_2["Life"][df_2.clusters==1],color="r")
plt.xlabel("K=2 chart")

plt.subplot(1,2,2)
plt.scatter(fuldf["GDP"][fuldf.Region=="Western Europe"],
            fuldf["Life"][fuldf.Region=="Western Europe"],color="r")

plt.scatter(fuldf["GDP"][fuldf.Region=="Latin America and Caribbean"],
            fuldf["Life"][fuldf.Region=="Latin America and Caribbean"],color="b")

plt.scatter(fuldf["GDP"][fuldf.Region=="Sub-Saharan Africa"],
            fuldf["Life"][fuldf.Region=="Sub-Saharan Africa"],color="g")
plt.xlabel("Real chart")
plt.show()




plt.subplot(1,2,1)
plt.scatter(df_2["GDP"][df_2.clusters==0],
            df_2["Life"][df_2.clusters==0],color="b")

plt.scatter(df_1["GDP"][df_1.clusters==1],
            df_1["Life"][df_1.clusters==1],color="g")

plt.scatter(df_1["GDP"][df_1.clusters==2],
            df_1["Life"][df_1.clusters==2],color="r")
plt.xlabel("K=3 chart")

plt.subplot(1,2,2)
plt.scatter(fuldf["GDP"][fuldf.Region=="Western Europe"],
            fuldf["Life"][fuldf.Region=="Western Europe"],color="r")

plt.scatter(fuldf["GDP"][fuldf.Region=="Latin America and Caribbean"],
            fuldf["Life"][fuldf.Region=="Latin America and Caribbean"],color="b")

plt.scatter(fuldf["GDP"][fuldf.Region=="Sub-Saharan Africa"],
            fuldf["Life"][fuldf.Region=="Sub-Saharan Africa"],color="g")
plt.xlabel("Real chart")
plt.show()

