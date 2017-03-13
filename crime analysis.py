from pandas import Series,DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import datetime
from pandas_datareader import data
from pandas_datareader import wb
import seaborn as sns
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import os

# Function to convert columns to numeric data
def convert_numeric(crime_data,col_name):
    crime_data[col_name]=crime_data[col_name].convert_objects(convert_numeric=True)
    return crime_data

# Function to format crime data    
def format_crime_data(crime_data):
    crime_data=crime_data.rename(columns=lambda x: x.strip())
    crime_data=crime_data.fillna(0)
    columns_list= crime_data.columns.tolist()
    for column in columns_list:
        if column not in (['State','County']):
            pd.to_numeric(column,errors='ignore')
    crime_data['Rape']=crime_data['Rape\n(revised\ndefinition)1']+crime_data['Rape\n(legacy\ndefinition)2']
    crime_data['sum_of_crimes']=crime_data['Violent\ncrime']+crime_data['Murder and\nnonnegligent\nmanslaughter']+\
    crime_data['Rape']+crime_data['Robbery']+crime_data['Aggravated\nassault']+crime_data['Property\ncrime']+\
    crime_data['Burglary']+crime_data['Larceny-\ntheft']+crime_data['Motor\nvehicle\ntheft']+crime_data['Arson3']
    return crime_data

# Function to add county code to crime and police data
def add_county_code(df):
    # format df to prepare for merge
    df['State']=df['State'].str.split('-').str.get(0).str.strip().str.lower()
    df['County']=df['County'].str.lower()+' county'
    
    # add state abbreviation to df based on State name
    ab_states=pd.read_csv('USStates.txt',header=None,names=['State','ab','area','pop'])
    ab_states['State']=ab_states['State'].str.lower()
    df=df.merge(ab_states)
    
    # add county code based on the state abb and county name
    county_code = pd.read_csv('national_county.txt',header=None, names=['ab','c1','c2','County','H'],dtype=str)
    county_code['code']=county_code['c1']+county_code['c2']
    county_code['County']=county_code['County'].str.lower()
    df=pd.merge(df, county_code, on=['County','ab'],how='left')
    return df


# read crime data and formating crime data
os.chdir('/Users/xinyingdu/Google Drive/UMN/Useful Code/Crime')
crime_data = pd.read_csv('CrimeDate2014.csv',skiprows = 4,skip_footer = 8,thousands=',',na_values=['0',0,' '])
crime_data = format_crime_data(crime_data)
crime_data = add_county_code(crime_data)

# read police data and formating police data
police_data = pd.read_csv('Police_2014.csv',skiprows=3,skip_footer=1,thousands=',')
police_data = add_county_code(police_data)
police_data = police_data.ix[:, ['code', 'Total law\nenforcement\nemployees']]
police_data.columns = ['code', 'police']

# read other predictor variables and save useful columns
population = pd.read_csv('PopulationEstimates.csv',usecols =['FIPS','POP_ESTIMATE_2014'], header=0,skiprows=2,thousands=',',dtype=str,na_values=['0',0,' '])
population.columns = ['code', 'population']

unem = pd.read_csv('Unemployment.csv',usecols =['FIPS_Code','Unemployment_rate_2014'], header=0,skiprows=6,thousands=',',dtype=str,na_values=['0',0,' '])
unem.columns = ['code', 'unem']

poverty = pd.read_csv('PovertyEstimates.csv',usecols = ['FIPStxt','POVALL_2014'],header=0,skiprows=2,thousands=',',dtype=str,na_values=['0',0,' '])
poverty.columns = ['code', 'poverty']

education = pd.read_csv('Education.csv',usecols=['FIPS Code','Percent of adults with less than a high school diploma, 2010-2014'],\
                        header=0,skiprows=4,thousands=',',dtype=str,na_values=['0',0,' '])
education.columns = ['code', 'education']

# merge all variables into dataframe crime
crime = pd.merge(crime_data, police_data, on='code', how='left')\
    .merge(population, on='code', how='left')\
    .merge(unem, on='code', how='left')\
    .merge(poverty, on='code', how='left')\
    .merge(education, on='code', how='left')

crime=crime.dropna()
crime=crime[crime['sum_of_crimes']!=0]

pop=crime['population'].str.replace(',','').astype(float)
edu=crime['education'].str.replace(',','').astype(float)
pov=crime['poverty'].str.replace(',','').astype(float)
"""
*************************************** Q1 Some Facts ***********************************************
1. List the top 3 states with the highest average criminal offense number?
2. List the top 3 states with the highest burglary crime rate? 
   Are they different with the top 3 states having the highest violent crime rate?
3. List Top 10 states with highest crime rate ?
"""
#1
all_by_state=crime.groupby('State').mean()['sum_of_crimes']
plt.figure(1)
plt.title('Average Criminal Offense Number')
all_by_state.plot(kind='bar')

#2
vio_by_state=DataFrame(crime.groupby('State').sum()['Violent\ncrime'])
vio_by_state['burglary']=crime.groupby('State').sum()['Burglary']

plt.title('Total Violent Criminal Offense Number')
vio_by_state.plot(kind='bar')

sum_by_state=crime.groupby('State').sum()['sum_of_crimes']
highest=sum_by_state[sum_by_state==sum_by_state.max()]
print ('{} has highest criminal activities that is {} in total in 2014.'.format(highest.index.values[0],highest[0]))

#3
to_sort=DataFrame(crime['sum_of_crimes'])
to_sort['State']=crime['State']
to_sort['county']=crime['code']
top10=to_sort.sort_values('sum_of_crimes',ascending=0).head(10)
result=top10.groupby('State').count()
max_count=result['sum_of_crimes'].max()
max_state=result[result['sum_of_crimes']==max_count].index.values[0]
print ('The state has most of the counties in the Crime Rate Top 10 List are from {}.'.format(max_state))

#6
plt.figure(3)
diff_to_sort=DataFrame(crime['State'])
diff_to_sort['Violent\ncrime']=crime['Violent\ncrime']
diff_to_sort['Rape']=crime['Rape']
diff_to_sort['diff']=(diff_to_sort['Violent\ncrime']-diff_to_sort['Rape']).abs()
diff_by_states=diff_to_sort.groupby('State',as_index=False).max()
diff_by_states=diff_by_states.sort_values(by='diff',ascending=0)
diff_by_states['diff'].head(10).plot(kind='bar').set_xticklabels(diff_by_states['State'].head(10))
plt.show()

"""
*************************************** Q2 Regresson ***********************************************
"""
# add log to crime
crime['sum_of_crimes_log']=crime['sum_of_crimes'].astype(float).apply(np.log10)
crime['police_log']=crime['police'].astype(float).apply(np.log10)
crime['population_log']=pop.apply(np.log10)
crime['poverty_log']=pov.apply(np.log10)
crime['education_log']=edu.apply(np.log10)

# perform regression
result = sm.ols(formula="sum_of_crimes_log ~ police_log+population_log+poverty_log+education_log", data=crime).fit()
print (result.params)
print (result.summary())

"""
Intercept         0.361705
police_log        0.827590
population_log   -0.190150
poverty_log       0.398847
education_log     0.171726
dtype: float64
                            OLS Regression Results                            
==============================================================================
Dep. Variable:      sum_of_crimes_log   R-squared:                       0.651
Model:                            OLS   Adj. R-squared:                  0.651
Method:                 Least Squares   F-statistic:                     919.7
Date:                Sat, 20 Aug 2016   Prob (F-statistic):               0.00
Time:                        18:12:25   Log-Likelihood:                -1044.9
No. Observations:                1973   AIC:                             2100.
Df Residuals:                    1968   BIC:                             2128.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Intercept          0.3617      0.170      2.126      0.034         0.028     0.695
police_log         0.8276      0.036     22.901      0.000         0.757     0.898
population_log    -0.1902      0.090     -2.108      0.035        -0.367    -0.013
poverty_log        0.3988      0.081      4.908      0.000         0.239     0.558
education_log      0.1717      0.064      2.677      0.007         0.046     0.298
==============================================================================
Omnibus:                      938.064   Durbin-Watson:                   1.445
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7625.542
Skew:                          -2.067   Prob(JB):                         0.00
Kurtosis:                      11.699   Cond. No.                         139.
==============================================================================
"""


"""
*************************************** Q3 Regression ***********************************************
"""

# read original crime data of California
crime_ca_data = pd.read_csv('Crime Ca 1985-2005.csv', skiprows = 4,index_col = 0)
crime_ca_data = crime_ca_data.T
crime_ca_data.index = crime_ca_data.index.astype(int)
crime_ca_data.columns = crime_ca_data.columns.map(lambda x : x.replace(' ','_')if isinstance(x,str)else x)
crime_ca_data = crime_ca_data.ix[:,'Violent_Crimes': 'Rape']

# population data
start = datetime.datetime(1985,1,1)
end = datetime.datetime(2015,1,1)
pop_data = data.DataReader("CAPOP", "fred", start, end).resample('A-DEC').mean()
pop_data.index = pop_data.index.year.astype(int)

# prison and parole data
prison = pd.read_csv('prisen_population.csv',index_col = 0)

# poverty data
poverty = pd.read_csv('poverty - ca.csv',index_col = 0)

# merge all ariables into dataframe crime_ca
crime_ca = pd.merge(crime_ca_data, prison, left_index=True, right_index = True)\
    .merge(pop_data, left_index=True, right_index = True)\
    .merge(poverty, left_index=True, right_index = True)

crime_ca = crime_ca.dropna()

# add log to crime
crime_ca['Violent_Crimes_log']=crime_ca['Violent_Crimes'].astype(float).apply(np.log10)
crime_ca['Rape_log']=crime_ca['Rape'].astype(float).apply(np.log10)
crime_ca['Homicide_log']=crime_ca['Homicide'].astype(float).apply(np.log10)

crime_ca['CAPOP_log']=crime_ca['CAPOP'].astype(float).apply(np.log10)
crime_ca['prison_log']=crime_ca['prison'].astype(float).apply(np.log10)
crime_ca['parole_log']=crime_ca['parole'].astype(float).apply(np.log10)
crime_ca['poverty_log']=crime_ca['poverty'].astype(float).apply(np.log10)

# perform regression
result2 = sm.ols(formula="Violent_Crimes_log ~ CAPOP_log+ parole_log+poverty_log", data=crime_ca).fit()
print (result2.params)
print (result2.summary())

'''
Intercept      18.049807
CAPOP_log      -3.817601
parole_log      0.524050
poverty_log     0.532549
dtype: float64
                            OLS Regression Results                            
==============================================================================
Dep. Variable:     Violent_Crimes_log   R-squared:                       0.806
Model:                            OLS   Adj. R-squared:                  0.779
Method:                 Least Squares   F-statistic:                     30.39
Date:                Tue, 23 Aug 2016   Prob (F-statistic):           5.25e-08
Time:                        12:46:45   Log-Likelihood:                 45.534
No. Observations:                  26   AIC:                            -83.07
Df Residuals:                      22   BIC:                            -78.03
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
Intercept      18.0498      1.612     11.200      0.000        14.707    21.392
CAPOP_log      -3.8176      0.450     -8.480      0.000        -4.751    -2.884
parole_log      0.5241      0.119      4.388      0.000         0.276     0.772
poverty_log     0.5325      0.172      3.101      0.005         0.176     0.889
==============================================================================
Omnibus:                        4.296   Durbin-Watson:                   0.563
Prob(Omnibus):                  0.117   Jarque-Bera (JB):                2.964
Skew:                          -0.817   Prob(JB):                        0.227
Kurtosis:                       3.260   Cond. No.                     1.44e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.44e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

result3 = sm.ols(formula="Rape_log ~ CAPOP_log+ parole_log+poverty_log", data=crime_ca).fit()
print (result3.params)
print (result3.summary())

'''
Intercept      10.606786
CAPOP_log      -1.615627
parole_log      0.142180
poverty_log    -0.000834
dtype: float64
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               Rape_log   R-squared:                       0.787
Model:                            OLS   Adj. R-squared:                  0.758
Method:                 Least Squares   F-statistic:                     27.09
Date:                Tue, 23 Aug 2016   Prob (F-statistic):           1.42e-07
Time:                        12:47:28   Log-Likelihood:                 58.595
No. Observations:                  26   AIC:                            -109.2
Df Residuals:                      22   BIC:                            -104.2
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
Intercept      10.6068      0.975     10.876      0.000         8.584    12.629
CAPOP_log      -1.6156      0.272     -5.931      0.000        -2.181    -1.051
parole_log      0.1422      0.072      1.967      0.062        -0.008     0.292
poverty_log    -0.0008      0.104     -0.008      0.994        -0.216     0.215
==============================================================================
Omnibus:                        0.755   Durbin-Watson:                   0.415
Prob(Omnibus):                  0.685   Jarque-Bera (JB):                0.532
Skew:                           0.337   Prob(JB):                        0.767
Kurtosis:                       2.807   Cond. No.                     1.44e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.44e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

result4 = sm.ols(formula="Homicide_log ~ CAPOP_log+ parole_log+poverty_log", data=crime_ca).fit()
print (result4.params)
print (result4.summary())

'''
Intercept      13.838339
CAPOP_log      -3.041258
parole_log      0.271507
poverty_log     0.536248
dtype: float64
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           Homicide_log   R-squared:                       0.617
Model:                            OLS   Adj. R-squared:                  0.565
Method:                 Least Squares   F-statistic:                     11.81
Date:                Tue, 23 Aug 2016   Prob (F-statistic):           8.13e-05
Time:                        12:48:15   Log-Likelihood:                 35.254
No. Observations:                  26   AIC:                            -62.51
Df Residuals:                      22   BIC:                            -57.48
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
Intercept      13.8383      2.393      5.783      0.000         8.875    18.801
CAPOP_log      -3.0413      0.669     -4.549      0.000        -4.428    -1.655
parole_log      0.2715      0.177      1.531      0.140        -0.096     0.639
poverty_log     0.5362      0.255      2.103      0.047         0.007     1.065
==============================================================================
Omnibus:                        1.889   Durbin-Watson:                   0.337
Prob(Omnibus):                  0.389   Jarque-Bera (JB):                1.681
Skew:                          -0.556   Prob(JB):                        0.431
Kurtosis:                       2.438   Cond. No.                     1.44e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.44e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

"""
************************************************************* Graphs *********************************************************
"""
#Violent_Crimes ~ CAPOP
fig, axes = plt.subplots(nrows=2, ncols=2)
crime_ca.plot.scatter(x='CAPOP',y='Violent_Crimes',ax=axes[0,0]); 
m, b = np.polyfit(crime_ca['CAPOP'], crime_ca['Violent_Crimes'], 1)
axes[0,0].set_title('Scatter between California population and violent crime data')
axes[0,0].plot(crime_ca['CAPOP'], m*crime_ca['CAPOP'] + b, '-g')

#Violent_Crimes_log ~ parole
crime_ca.plot.scatter(x='parole',y='Violent_Crimes',ax=axes[1,0]); 
m, b = np.polyfit(crime_ca['parole'], crime_ca['Violent_Crimes'], 1)
axes[1,0].set_title('Scatter between California parole volumn and violent crime data')
axes[1,0].plot(crime_ca['parole'], m*crime_ca['parole'] + b, '-g')

#Violent_Crimes_log ~ poverty
crime_ca.plot.scatter(x='poverty',y='Violent_Crimes',ax=axes[0,1]); 
m, b = np.polyfit(crime_ca['poverty'], crime_ca['Violent_Crimes'], 1)
axes[0,1].set_title('Scatter between California poverty volumn and violent crime data')
axes[0,1].plot(crime_ca['poverty'], m*crime_ca['poverty'] + b, '-g')

#Violent_Crimes_log ~ prison
crime_ca.plot.scatter(x='prison',y='Violent_Crimes',ax=axes[1,1]); 
m, b = np.polyfit(crime_ca['prison'], crime_ca['Violent_Crimes'], 1)
axes[1,1].set_title('Scatter between California prison and violent crime data')
axes[1,1].plot(crime_ca['prison'], m*crime_ca['prison'] + b, '-g')
plt.show()

#Q2 sum_of_crimes ~ education
plt.figure()
crime[['education']] = crime[['education']].apply(pd.to_numeric)
f1=plt.subplot(231)
m, b = np.polyfit(crime['education'], crime['sum_of_crimes'], 1)
crime.plot.scatter(x='education',y='sum_of_crimes',ax=f1)
f1.set_title('Education Level')
f1.plot(crime['education'], m*crime['education'] + b, '-g')
#sns.regplot(x="education", y="sum_of_crimes", data=crime)

#Q2 sum_of_crimes ~ poverty
f2=plt.subplot(232)
crime["poverty"] = crime["poverty"].str.replace(r'[,]', '').astype('float')
m, b = np.polyfit(crime['poverty'], crime['sum_of_crimes'], 1)
crime.plot.scatter(x='poverty',y='sum_of_crimes',ax=f2)
f2.set_title('Poverty Level')
f2.plot(crime['poverty'], m*crime['poverty'] + b, '-g')

#Q2 sum_of_crimes ~ population
f3=plt.subplot(233)
crime["population"] = crime["population"].str.replace(r'[,]', '').astype('float')
m, b = np.polyfit(crime['population'], crime['sum_of_crimes'], 1)
crime.plot.scatter(x='population',y='sum_of_crimes',ax=f3)
f3.set_title('Population')
f3.plot(crime['population'], m*crime['population'] + b, '-g')

#Q2 sum_of_crimes ~ police
f4=plt.subplot(234)
m, b = np.polyfit(crime['police'], crime['sum_of_crimes'], 1)
crime.plot.scatter(x='police',y='sum_of_crimes',ax=f4)
f4.set_title('Police')
f4.plot(crime['police'], m*crime['police'] + b, '-g')

#Q2 sum_of_crimes ~ unemployment
f5=plt.subplot(235)
crime["unem"] = crime["unem"].str.replace(r'[,]', '').astype('float')
m, b = np.polyfit(crime['unem'], crime['sum_of_crimes'], 1)
crime.plot.scatter(x='unem',y='sum_of_crimes',ax=f5)
f5.set_title('Umemployment')
f5.plot(crime['unem'], m*crime['unem'] + b, '-g')
plt.show()

"""
************************************** Extra: Mapping states have top crime rates on Maps ********************************************
"""
plt.figure()
# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# draw state boundaries.
# data from U.S Census Bureau
# http://www.census.gov/geo/www/cob/st2000.html
shp_info = m.readshapefile('states','states',drawbounds=True)

# population density by state from
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax = 450 # set range.
print(m.states_info[0].keys())
for shapedict in m.states_info:
    statename = shapedict['STATE_NAME']
    statenames.append(statename)

# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
seg = m.states[statenames.index('Texas')]
poly = Polygon(seg, facecolor='red',edgecolor='red')
ax.add_patch(poly)
seg = m.states[statenames.index('Florida')]
poly = Polygon(seg, facecolor='red',edgecolor='red')
ax.add_patch(poly)
seg = m.states[statenames.index('Georgia')]
poly = Polygon(seg, facecolor='red',edgecolor='red')
ax.add_patch(poly)
seg = m.states[statenames.index('Louisiana')]
poly = Polygon(seg, facecolor='red',edgecolor='red')
ax.add_patch(poly)
seg = m.states[statenames.index('South Carolina')]
poly = Polygon(seg, facecolor='orange',edgecolor='orange')
ax.add_patch(poly)
seg = m.states[statenames.index('Maryland')]
poly = Polygon(seg, facecolor='orange',edgecolor='orange')
ax.add_patch(poly)
seg = m.states[statenames.index('North Carolina')]
poly = Polygon(seg, facecolor='orange',edgecolor='orange')
ax.add_patch(poly)
seg = m.states[statenames.index('Virginia')]
poly = Polygon(seg, facecolor='orange',edgecolor='orange')
ax.add_patch(poly)
seg = m.states[statenames.index('Tennessee')]
poly = Polygon(seg, facecolor='yellow',edgecolor='yellow')
ax.add_patch(poly)
seg = m.states[statenames.index('Washington')]
poly = Polygon(seg, facecolor='yellow',edgecolor='yellow')
ax.add_patch(poly)
seg = m.states[statenames.index('Montana')]
poly = Polygon(seg, facecolor='yellow',edgecolor='yellow')
ax.add_patch(poly)

plt.show()
