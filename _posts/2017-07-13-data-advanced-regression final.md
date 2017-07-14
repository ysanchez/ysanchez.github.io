---
layout: post
title: Advanced Regression Housing Predicion
---

# First Notebook
**I combined the train and test data set so the columns remain equal and to keep control of the changes. You want to make sure that the exact changes made in the train dataset is made in the test dataset.**


```python
import pandas as pd
import numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
frames = [train, test]
data = pd.concat(frames)
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>Alley</th>
      <th>BedroomAbvGr</th>
      <th>BldgType</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType</th>
      <th>ScreenPorch</th>
      <th>Street</th>
      <th>TotRmsAbvGrd</th>
      <th>TotalBsmtSF</th>
      <th>Utilities</th>
      <th>WoodDeckSF</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>No</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>8</td>
      <td>856.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2003</td>
      <td>2003</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Gd</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>1262.0</td>
      <td>AllPub</td>
      <td>298</td>
      <td>1976</td>
      <td>1976</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Mn</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>920.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2001</td>
      <td>2002</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>Gd</td>
      <td>No</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>7</td>
      <td>756.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>1915</td>
      <td>1970</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Av</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>9</td>
      <td>1145.0</td>
      <td>AllPub</td>
      <td>192</td>
      <td>2000</td>
      <td>2000</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



**There are many columns in this data set resulting in a lot of information. The list of information provides major detail such as what type of information each column contains. This valuable information helps us when working with the data, but telling what code will be needed to make the data useful in determining housing price.** 


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2919 entries, 0 to 1458
    Data columns (total 81 columns):
    1stFlrSF         2919 non-null int64
    2ndFlrSF         2919 non-null int64
    3SsnPorch        2919 non-null int64
    Alley            198 non-null object
    BedroomAbvGr     2919 non-null int64
    BldgType         2919 non-null object
    BsmtCond         2837 non-null object
    BsmtExposure     2837 non-null object
    BsmtFinSF1       2918 non-null float64
    BsmtFinSF2       2918 non-null float64
    BsmtFinType1     2840 non-null object
    BsmtFinType2     2839 non-null object
    BsmtFullBath     2917 non-null float64
    BsmtHalfBath     2917 non-null float64
    BsmtQual         2838 non-null object
    BsmtUnfSF        2918 non-null float64
    CentralAir       2919 non-null object
    Condition1       2919 non-null object
    Condition2       2919 non-null object
    Electrical       2918 non-null object
    EnclosedPorch    2919 non-null int64
    ExterCond        2919 non-null object
    ExterQual        2919 non-null object
    Exterior1st      2918 non-null object
    Exterior2nd      2918 non-null object
    Fence            571 non-null object
    FireplaceQu      1499 non-null object
    Fireplaces       2919 non-null int64
    Foundation       2919 non-null object
    FullBath         2919 non-null int64
    Functional       2917 non-null object
    GarageArea       2918 non-null float64
    GarageCars       2918 non-null float64
    GarageCond       2760 non-null object
    GarageFinish     2760 non-null object
    GarageQual       2760 non-null object
    GarageType       2762 non-null object
    GarageYrBlt      2760 non-null float64
    GrLivArea        2919 non-null int64
    HalfBath         2919 non-null int64
    Heating          2919 non-null object
    HeatingQC        2919 non-null object
    HouseStyle       2919 non-null object
    Id               2919 non-null int64
    KitchenAbvGr     2919 non-null int64
    KitchenQual      2918 non-null object
    LandContour      2919 non-null object
    LandSlope        2919 non-null object
    LotArea          2919 non-null int64
    LotConfig        2919 non-null object
    LotFrontage      2433 non-null float64
    LotShape         2919 non-null object
    LowQualFinSF     2919 non-null int64
    MSSubClass       2919 non-null int64
    MSZoning         2915 non-null object
    MasVnrArea       2896 non-null float64
    MasVnrType       2895 non-null object
    MiscFeature      105 non-null object
    MiscVal          2919 non-null int64
    MoSold           2919 non-null int64
    Neighborhood     2919 non-null object
    OpenPorchSF      2919 non-null int64
    OverallCond      2919 non-null int64
    OverallQual      2919 non-null int64
    PavedDrive       2919 non-null object
    PoolArea         2919 non-null int64
    PoolQC           10 non-null object
    RoofMatl         2919 non-null object
    RoofStyle        2919 non-null object
    SaleCondition    2919 non-null object
    SalePrice        1460 non-null float64
    SaleType         2918 non-null object
    ScreenPorch      2919 non-null int64
    Street           2919 non-null object
    TotRmsAbvGrd     2919 non-null int64
    TotalBsmtSF      2918 non-null float64
    Utilities        2917 non-null object
    WoodDeckSF       2919 non-null int64
    YearBuilt        2919 non-null int64
    YearRemodAdd     2919 non-null int64
    YrSold           2919 non-null int64
    dtypes: float64(12), int64(26), object(43)
    memory usage: 1.8+ MB


**There are many variables in each column. It is important to review each column and look at its variables.**


```python
pd.value_counts(data['BldgType'].values, sort=True)
```




    1Fam      2425
    TwnhsE     227
    Duplex     109
    Twnhs       96
    2fmCon      62
    dtype: int64



**There are many basement columns, not all were needed for this project** (my opinion)


```python
pd.value_counts(data['BsmtFinType1'].values, sort=True)

```




    Unf    851
    GLQ    849
    ALQ    429
    Rec    288
    BLQ    269
    LwQ    154
    dtype: int64




```python
pd.value_counts(data['BsmtCond'].values, sort=True)
```




    TA    2606
    Gd     122
    Fa     104
    Po       5
    dtype: int64




```python
data = data.drop('BsmtExposure')
data = data.drop('BsmtQual')
```

**Growing up Southern California Central Air is very important. I actually have it on as i write this.**


```python
pd.value_counts(data['CentralAir'].values, sort=True)
data['CentralAir'] = pd.get_dummies(data['CentralAir'], drop_first=True)
```


```python
pd.value_counts(data['Condition1'].values, sort=True)
```




    Norm      2511
    Feedr      164
    Artery      92
    RRAn        50
    PosN        39
    RRAe        28
    PosA        20
    RRNn         9
    RRNe         6
    dtype: int64




```python
pd.value_counts(data['Electrical'].values, sort=True)
```




    SBrkr    2671
    FuseA     188
    FuseF      50
    FuseP       8
    Mix         1
    dtype: int64




```python
pd.value_counts(data['ExterCond'].values, sort=True)
data = data.drop('ExterQual')
```


```python
pd.value_counts(data['Exterior1st'])
```




    VinylSd    1025
    MetalSd     450
    HdBoard     442
    Wd Sdng     411
    Plywood     221
    CemntBd     126
    BrkFace      87
    WdShing      56
    AsbShng      44
    Stucco       43
    BrkComm       6
    AsphShn       2
    Stone         2
    CBlock        2
    ImStucc       1
    Name: Exterior1st, dtype: int64




```python
pd.value_counts(data['Fence'].values, sort=True)
```




    MnPrv    329
    GdPrv    118
    GdWo     112
    MnWw      12
    dtype: int64




```python
pd.value_counts(data['Functional'].values, sort=True)
```




    Typ     2717
    Min2      70
    Min1      65
    Mod       35
    Maj1      19
    Maj2       9
    Sev        2
    dtype: int64



**There were mulitple columns for garage. Instead of using every single one I decided to engineer the variables to 'yes' and 'no'**


```python
pd.value_counts(data['GarageCond'].values, sort=True)
data['GarageCond'] = data['GarageCond'].fillna('No')
data['GarageCond'] = data['GarageCond'].apply(lambda x: x.replace(x, 'Yes') if x != 'No' else x)
pd.value_counts(data['GarageCond'].values, sort=True)
```




    Yes    2760
    No      159
    dtype: int64




```python
data= data.drop('GarageFinish')
```


```python
pd.value_counts(data['Heating'].values, sort=True)
```




    GasA     2874
    GasW       27
    Grav        9
    Wall        6
    OthW        2
    Floor       1
    dtype: int64




```python
pd.value_counts(data['KitchenQual'].values, sort=False)
```




    Gd    1151
    Ex     205
    Fa      70
    TA    1492
    dtype: int64




```python
pd.value_counts(data['LandContour'].values, sort=True)
```




    Lvl    2622
    HLS     120
    Bnk     117
    Low      60
    dtype: int64



**When looking at the variables there were only 10 homes in the data set that had a pool so I decided engineer the variables to 'yes' and 'no'**


```python
pd.value_counts(data['PoolQC'].values, sort=True)
data['PoolQC']= data['PoolQC'].fillna('No')
data['PoolQC'] = data['PoolQC'].apply(lambda x: x.replace(x, 'Yes') if x == 'Ex' or  x == 'Gd' or x == 'Fa' else x)
pd.value_counts(data['PoolQC'].values, sort=True)
```




    No     2909
    Yes      10
    dtype: int64




```python
pd.value_counts(data['RoofMatl'].values, sort=True)
```




    CompShg    2876
    Tar&Grv      23
    WdShake       9
    WdShngl       7
    Membran       1
    ClyTile       1
    Metal         1
    Roll          1
    dtype: int64




```python
pd.value_counts(data['SaleCondition'].values, sort=True)
```




    Normal     2402
    Partial     245
    Abnorml     190
    Family       46
    Alloca       24
    AdjLand      12
    dtype: int64




```python
pd.value_counts(data['Utilities'].values, sort=True)
```




    AllPub    2916
    NoSeWa       1
    dtype: int64



**For this data set it important to create dummies. All variables must be numbers in order for the algorithm to work.**


For dummies to work, it is important not to create dummies for each single column. A mulitpude of columns must be created with values of 1 or 0 so one variable does not supercede another. 


```python
data = pd.get_dummies(data, drop_first=True)

```

**Some columns are very valuable, but there physical value is small compared to others. In this case specific clumns were raised to the second power to give them more weight.**


```python
data= data.drop('Functional')
data['BedroomAbvGr'] = pow(data['BedroomAbvGr'], 2)
data['BsmtFullBath'] = pow(data['BsmtFullBath'], 2)
data['BsmtHalfBath'] = pow(data['BsmtHalfBath'], 2)
data['FullBath'] = pow(data['FullBath'], 2)
data['Fireplaces'] = pow(data['Fireplaces'], 2)
data['GarageCars'] = pow(data['GarageCars'], 2)
data['SaleType_New'] = pow(data['SaleType_New'], 2)
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>BedroomAbvGr</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtUnfSF</th>
      <th>CentralAir</th>
      <th>...</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>Street_Pave</th>
      <th>Utilities_NoSeWa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>9</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>284.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>9</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>9</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>16</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 243 columns</p>
</div>



**Once the data is ready we have to separate the data to what was the original train and test data set. I renamed it to be used in the notebook where the algorthm is used to produce the prediction.**


```python
train = data.iloc[:1460]
test = data.iloc[1460:]
```

**I created a new file named "train data" which is a new .csv file that folds the new train dataset.**


```python
train.info()
train = train.fillna(0)
train.to_csv('train data.csv')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1460 entries, 0 to 1459
    Columns: 243 entries, 1stFlrSF to Utilities_NoSeWa
    dtypes: float64(12), int64(26), uint8(205)
    memory usage: 737.1 KB


**I created a second file which hold the test data set, which is also saved in a .csv.**


```python
test.info()
test=test.fillna(0)
test.to_csv('test_data.csv')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1459 entries, 0 to 1458
    Columns: 243 entries, 1stFlrSF to Utilities_NoSeWa
    dtypes: float64(12), int64(26), uint8(205)
    memory usage: 736.6 KB



```python

```
