#Importing necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes =True)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import glob

#splitting bigger csv files into chunks
chunksize = 10 ** 6
batch_no = 1
for chunk in pd.read_csv('/home/subarna/Documents/BMTC/BMTC/train.csv',chunksize=chunksize):
    chunk.to_csv('training set'+str(batch_no)+'.csv',index=False)
    batch_no += 1

#removing defective informations
number_of_files = 215

for i in range(1, number_of_files+1):
    data = pd.read_csv("/home/subarna/Documents/BMTC/BMTC/training_sets/training set{}.csv".format(i))
    del data['27.00']
    del data['214.0']
    data.set_axis(['Bus_id','Latitude','Longitude','Date and Time'], axis=1, inplace=True)
    data.drop(data[data['Latitude'] < 12].index, inplace = True)
    data.drop(data[data['Latitude'] >14].index, inplace = True)
    data.drop(data[data['Longitude'] <77].index, inplace = True)
    data.drop(data[data['Longitude'] >78.5].index, inplace = True)
    data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    data.to_csv('mod_training_set'+str(i)+'.csv',index=False)

#removing duplicate cells
number_of_files = 215

for i in range(1, number_of_files+1):
    data = pd.read_csv("/home/subarna/Documents/BMTC/BMTC/mod_data/mod_training_set{}.csv".format(i))
    data['Latitude'] = data['Latitude'].map(str) + ':' +data['Longitude'].map(str)
    data=data.drop(['Longitude'], axis=1).rename(columns={'Latitude':'LATLONG'})
    data.drop_duplicates(subset =["LATLONG",'Bus_id'],keep = 'first', inplace = True)
    data.to_csv('set'+str(i)+'.csv',index=False)

#rename date and time

number_of_files = 37
for i in range(1, number_of_files+1):
    df = pd.read_csv("/home/subarna/Documents/BMTC/new_task/date1/set{}.csv".format(i))
    df.rename(columns={'Date and Time': 'Date_Time'}, inplace=True)
    df[['Date','Time']] = df.Date_Time.str.split(expand=True)
    del df['Date_Time']
    for j, x in df.groupby('Date'):
        x.to_csv(str(j)+"_{}.csv".format(i),index=False)

for i in range(39,74):
    df = pd.read_csv("/home/subarna/Documents/BMTC/new_task/date2/set{}.csv".format(i))
    df.rename(columns={'Date and Time': 'Date_Time'}, inplace=True)
    df[['Date','Time']] = df.Date_Time.str.split(expand=True)
    del df['Date_Time']
    for j, x in df.groupby('Date'):
        x.to_csv(str(j)+"_{}.csv".format(i),index=False)

for i in range(75,109):
    df = pd.read_csv("/home/subarna/Documents/BMTC/new_task/date3/set{}.csv".format(i))
    df.rename(columns={'Date and Time': 'Date_Time'}, inplace=True)
    df[['Date','Time']] = df.Date_Time.str.split(expand=True)
    del df['Date_Time']
    for j, x in df.groupby('Date'):
        x.to_csv(str(j)+"_{}.csv".format(i),index=False)

for i in range(110,145):
    df = pd.read_csv("/home/subarna/Documents/BMTC/new_task/date4/set{}.csv".format(i))
    df.rename(columns={'Date and Time': 'Date_Time'}, inplace=True)
    df[['Date','Time']] = df.Date_Time.str.split(expand=True)
    del df['Date_Time']
    for j, x in df.groupby('Date'):
        x.to_csv(str(j)+"_{}.csv".format(i),index=False)

for i in range(146,181):
    df = pd.read_csv("/home/subarna/Documents/BMTC/new_task/date5/set{}.csv".format(i))
    df.rename(columns={'Date and Time': 'Date_Time'}, inplace=True)
    df[['Date','Time']] = df.Date_Time.str.split(expand=True)
    del df['Date_Time']
    for j, x in df.groupby('Date'):
        x.to_csv(str(j)+"_{}.csv".format(i),index=False)

for i in range(182,216):
    df = pd.read_csv("/home/subarna/Documents/BMTC/new_task/date6/set{}.csv".format(i))
    df.rename(columns={'Date and Time': 'Date_Time'}, inplace=True)
    df[['Date','Time']] = df.Date_Time.str.split(expand=True)
    del df['Date_Time']
    for j, x in df.groupby('Date'):
        x.to_csv(str(j)+"_{}.csv".format(i),index=False)

# get data file names
path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date1_mod'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('date1.csv',index=False)


path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date2_mod'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('date2.csv',index=False)


path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date3_mod'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('date3.csv',index=False)

path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date4_mod'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('date4.csv',index=False)


path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date5_mod'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('date5.csv',index=False)

path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date6_mod'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('date6.csv',index=False)


#reading file with particular date
df=pd.read_csv("/home/sandipansinha22/Documents/date1.csv")


#splitting individual date file into many as per bus id
for i,x in df.groupby('Bus_id'):
    x.to_csv("{}.csv".format(i),index=False)


df=pd.read_csv('/home/subarna/Documents/BMTC/new_task/modifie_dates/date1/150218000.csv')
df_sub=df.sample(n=100)
df
df[['lat','long']] = df.LATLONG.str.split(":",expand=True)
df
del df['LATLONG']
df
df['Date'] = df['Date'].map(str)+' '+df['Time'].map(str)
df=df.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
df
df['Time_stamp']=pd.to_datetime(df.Time_stamp)
df
df.dtypes
cols = df.select_dtypes(include=['object']).columns
df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
df
df.dtypes
len(df['Time_stamp'])



path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date1'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


path ='/home/subarna/Documents/BMTC/new_task/modifie_dates/date1'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


path ='/home/subarna/Documents/BMTC/date2_split'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


path ='/home/subarna/Documents/BMTC/date3_split'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


path ='/home/subarna/Documents/BMTC/date4_split'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


path ='/home/subarna/Documents/BMTC/date5_split'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


path ='/home/subarna/Documents/BMTC/date6_split'
filenames = glob.glob(path + "/*.csv")
i=1
for filename in filenames:
    df=pd.read_csv(filename)
    q=15
    for x in range(1,q+1):
        if df.shape[0]>=100:
            df_sub=df.sample(n=100)
        else:
            df_sub=df.sample(n=100,replace=True)
        df1=df_sub.sort_values(by=['Time'])
        df1['Date'] = df1['Date'].map(str)+' ' +df1['Time'].map(str)
        df1=df1.drop(['Time'], axis=1).rename(columns={'Date':'Time_stamp'})
        df1['Time_stamp']=pd.to_datetime(df1.Time_stamp)
        u=[df1.Time_stamp.max()-df1.Time_stamp.min()]
        df0=df1.groupby('Bus_id')['LATLONG'].apply(' '.join).reset_index()
        df3 = df0['LATLONG'].str.split(n=100, expand=True)
        df3.columns = ['LATLONG{}'.format(x+1) for x in df3.columns]
        df0 = df0.join(df3)
        del df0['LATLONG']
        df0=df0.rename(columns={'Bus_id':'Id'})
        dk = pd.DataFrame({'A':u})
        dk['B'] = (pd.Timestamp('now').normalize() + dk['A']).dt.time
        df0.insert(1,'TimeStamp',dk['B'],True)
        df0.insert(2,'Date','2016-07-01',True)
        df0['TimeStamp'] = df0['Date'].map(str)+' '+df0['TimeStamp'].map(str)
        del df0['Date']
        df0['TimeStamp']=pd.to_datetime(df0.TimeStamp)
        df0.to_csv('set_'+str(i)+'.csv',index=False)
        i+=1


# get data file names
path ='/home/subarna/Documents/BMTC/trips'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('f1.csv',index=False)


# get data file names
path ='/home/subarna/Documents/BMTC/trip2'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('f2.csv',index=False)


# get data file names
path ='/home/subarna/Documents/BMTC/trip3'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('f3.csv',index=False)


# get data file names
path ='/home/subarna/Documents/BMTC/trip4'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('f4.csv',index=False)


# get data file names
path ='/home/subarna/Documents/BMTC/trip5'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('f5.csv',index=False)


# get data file names
path ='/home/subarna/Documents/BMTC/trip6'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('f6.csv',index=False)


#merging all trips
path ='/home/subarna/Documents/BMTC/final training data'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.to_csv('train_data.csv',index=False)


for x in range(1,7):
    df=pd.read_csv('f{}.csv'.format(x))
    df[['Date','Time']] = df.TimeStamp.str.split(expand=True)
    del df['TimeStamp']
    del df['Date']
    df[['HH','MM','SS']] = df.Time.str.split(":",n=3,expand=True)
    del df['Time']
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    convert_dict = {'HH':int,'MM':int,'SS':int} 
    df = df.astype(convert_dict)
    df['duration']=df['HH']*3600+df['MM']*60+df['SS']
    del df['HH']
    del df['MM']
    del df['SS']
    for i in range(1,101):
        df[['lat{}'.format(i),'long{}'.format(i)]]=df['LATLONG{}'.format(i)].str.split(":", n = 1, expand = True)
        del df['LATLONG{}'.format(i)]
    cols = df.select_dtypes(include=['object']).columns
    df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    df.to_csv('final{}.csv'.format(x))


df=pd.read_csv('data.csv')
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df.shape
df.drop_duplicates(keep=False,inplace=True)
df.shape
df.head()
#new histogram plots
#histogram of column Pregnancies
fig=plt.figure(figsize=(5000,10))
df.hist(column="duration")
plt.xlabel("duration",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.drop(df[df['duration'] < 10000].index, inplace = True)
df.shape
#new histogram plots
#histogram of column Pregnancies
fig=plt.figure(figsize=(5000,10))
df.hist(column="duration")
plt.xlabel("duration",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
# set the background style of the plot 
sns.set_style('whitegrid') 
sns.distplot(df['duration'], kde = False, color ='red', bins = 100) 


df.to_csv('f_data.csv')
df=pd.read_csv('f_data.csv')
df.head()
del df['Unnamed: 0']
df.head()

X=df.drop("duration",axis=1)
y=df.duration
df.dtypes

#Random_Forest
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1,random_state=0)
regressor.fit(X,y)
from sklearn.externals import joblib
joblib.dump(regressor,'model.pkl')
y_pred = regressor.predict(X)
print(y_pred)
print(regressor.best_score_)

regressor.score(X, y)
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
joblib.dump(regressor,'model2.pkl')
regressor.score(X, y)
test=pd.read_csv('/home/subarna/Documents/final_bmtc_test_data.csv')
test.head()
del test['TimeStamp']

test=test.rename(columns={"BusId":"Id"})
for x in range(1,101):
    test[['lat{}'.format(x),'long{}'.format(x)]]=test['LATLONG{}'.format(x)].str.split(":", n = 1, expand = True)
    del test['LATLONG{}'.format(x)]
test.head()
test.shape
load_model=joblib.load("model2.pkl")

y_pred = load_model.predict(test)

data = np.array(y_pred) 
  
# creating series 
s = pd.Series(data) 
print(s) 

s.to_csv("results.csv")

#Linear Regression
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(x,y)


y_pred = load_model.predict(test)

data = np.array(y_pred) 
  
# creating series 
s = pd.Series(data) 
print(s) 

s.to_csv("result_lin.csv")


