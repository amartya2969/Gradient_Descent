import pandas as pd
df=pd.read_csv('3D_spatial_network.txt',sep=",",header=None)
df.to_csv('data.csv',index=False)
df1=pd.read_csv('data.csv')
first_column=df1.columns[0]
df1.drop([first_column],axis=1, inplace=True)
df1.to_csv('data.csv',index=False,header=None)
print(df1.shape[0])
