import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Travel.csv")
# print(df.head(10))

# print(df.info())

# print(df.duplicated().sum())

# print(df.columns)

# print(df.Gender.unique())
df['Gender'] = df['Gender'].replace(to_replace={'Fe Male':'Female'})
# print(df['Gender'].unique())

# print(df.isnull().sum().sort_values(ascending=False))
# print(round(100*(df.isnull().sum() / len(df.index)) , 2))
# df = df.dropna(axis = 0)
# print(round(100*(df.isnull().sum() / len(df.index)) , 2))

# cats = ['']
# print("categorical data",cats)

# nums = []
# print("numerical data",nums)

# for col in cats:
#     print(f"{col} has {df[col].unique()} values \n")


# for col in nums:
#     print(f"{col} has {df[col].unique()} values \n")

# for cat_columns in cats:
#     plt.figure(figsize = (8,5))
#     sns.countplot(x = cat_columns , data = df , palette='viridis')
#     plt.title(f'univariate analysis - {cat_columns}')
#     plt.show()

# for num_columns in nums:
#     plt.figure(figsize=(8,5))
#     sns.histplot(df[num_columns] , kde= True , color = 'skyblue')
#     plt.title(f'univarate analysis - {num_columns}')
#     plt.show()

# plt.hist(df['Age'].dropna() , bins = 30 , color = 'skyblue')
# plt.show()

# sns.countplot(x = 'TypeofContact' ,data = df,palette='viridis')
# plt.title('Different contact status')
# plt.xlabel('contact levels')
# plt.ylabel('no.of contacts')
# plt.show()

# print(df[cats].describe().T)

# bivariate analysis

# sns.scatterplot(x = 'Age' , y = 'DurationOfPitch' , data = df , hue = 'ProdTaken' , palette='coolwarm')
# plt.show()

# cross_tab = pd.crosstab(df['MaritalStatus'] , df['ProdTaken'] , normalize = 'index')
# cross_tab.plot(kind = 'bar' , stacked = True , color = ['skyblue' , 'yellow'])
# plt.show()

# sns.barplot(x = 'ProductPitched' , y = 'PitchSatisfactionScore' , data = df , palette='coolwarm')
# plt.show()

# sns.lineplot(x = 'NumberOfFollowups' , y = 'PitchSatisfactionScore' , data = df , palette='coolwarm' , marker = 'o')
# plt.show()

# mari_cont = pd.crosstab(df['MaritalStatus'] , df['TypeofContact'])
# mari_cont.plot(stacked=True , kind = 'bar')
# plt.show()

# sns.violinplot(x = 'ProdTaken' , y = 'Age' , data = df , hue ='OwnCar' , palette='viridis')
# plt.show()

# sns.swarmplot(x = 'ProdTaken' , y = 'Age' , data = df , hue ='OwnCar' , palette='viridis')
# plt.show()

from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111 , projection ='3d')
# ax.scatter(df['Age'] , df['MonthlyIncome'] , df['DurationOfPitch'] , c = df['ProdTaken'] , cmap = 'viridis')
# ax.set_xlabel('Age')
# ax.set_ylabel('Monthly Income')
# ax.set_zlabel('3D plot')
# plt.show()

# numeric_cols = df.select_dtypes(include = 'number')
# sns.heatmap(numeric_cols.corr() , annot = True , fmt = '.3f')
# plt.show()

# feature engineering

# df =df.drop('CustomerID' , axis = 1)
# print(df.columns)

nums = [col for col in df.columns if df[col].dtype != 'object']
print("length of numerical data",len(nums))

cats = [col for col in df.columns if df[col].dtype == 'object']
print(" length of categorical data",len(cats))

from sklearn.model_selection import train_test_split
x = df.drop('ProdTaken' ,axis = 1)
y = df['ProdTaken']

x_tain ,x_test,y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)
# print(x_tain ,x_test,y_train , y_test)
# print(x_tain.shape ,x_test.shape,)
# print(y_train.shape , y_test.shape)

