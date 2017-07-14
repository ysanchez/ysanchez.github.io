
# Second Notebook
**I took the dataset from the first notebook that I produced. In this notebook I only work with one data set at a time. I start with the train data set first.**


```python
import pandas as pd
import numpy as np

train = pd.read_csv('train data.csv')
train = train.drop('Unnamed: 0', 1)
train =train.drop('Id', 1)
train.head()
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
<p>5 rows × 242 columns</p>
</div>



**Running corr() helps you understand a bit better how one variables correlates to another.**


```python
train.corr()
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
      <th>1stFlrSF</th>
      <td>1.000000</td>
      <td>-0.202646</td>
      <td>0.056104</td>
      <td>0.136324</td>
      <td>0.445863</td>
      <td>0.097117</td>
      <td>0.238363</td>
      <td>0.002471</td>
      <td>0.317987</td>
      <td>0.146953</td>
      <td>...</td>
      <td>0.033381</td>
      <td>0.007559</td>
      <td>-0.011789</td>
      <td>0.006094</td>
      <td>-0.043721</td>
      <td>0.221219</td>
      <td>-0.008215</td>
      <td>-0.198056</td>
      <td>0.005950</td>
      <td>0.012287</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>-0.202646</td>
      <td>1.000000</td>
      <td>-0.024358</td>
      <td>0.497711</td>
      <td>-0.137079</td>
      <td>-0.099260</td>
      <td>-0.152625</td>
      <td>-0.029120</td>
      <td>0.004469</td>
      <td>-0.011803</td>
      <td>...</td>
      <td>0.007628</td>
      <td>0.003778</td>
      <td>-0.018808</td>
      <td>0.016175</td>
      <td>0.012602</td>
      <td>0.010810</td>
      <td>-0.036082</td>
      <td>0.026769</td>
      <td>0.046983</td>
      <td>-0.020818</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>0.056104</td>
      <td>-0.024358</td>
      <td>1.000000</td>
      <td>-0.026945</td>
      <td>0.026451</td>
      <td>-0.029993</td>
      <td>-0.004511</td>
      <td>0.029822</td>
      <td>0.020764</td>
      <td>0.030692</td>
      <td>...</td>
      <td>-0.006098</td>
      <td>-0.004309</td>
      <td>-0.009162</td>
      <td>-0.006820</td>
      <td>-0.006820</td>
      <td>0.019596</td>
      <td>-0.005279</td>
      <td>-0.014211</td>
      <td>0.007473</td>
      <td>-0.003046</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>0.136324</td>
      <td>0.497711</td>
      <td>-0.026945</td>
      <td>1.000000</td>
      <td>-0.087304</td>
      <td>-0.019908</td>
      <td>-0.111115</td>
      <td>0.045787</td>
      <td>0.149744</td>
      <td>-0.021231</td>
      <td>...</td>
      <td>0.030047</td>
      <td>-0.028730</td>
      <td>-0.013874</td>
      <td>0.081051</td>
      <td>-0.029071</td>
      <td>-0.054679</td>
      <td>0.022234</td>
      <td>0.046933</td>
      <td>0.030573</td>
      <td>0.000620</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>0.445863</td>
      <td>-0.137079</td>
      <td>0.026451</td>
      <td>-0.087304</td>
      <td>1.000000</td>
      <td>-0.050117</td>
      <td>0.611558</td>
      <td>0.060164</td>
      <td>-0.495251</td>
      <td>0.166468</td>
      <td>...</td>
      <td>0.008951</td>
      <td>0.030694</td>
      <td>-0.021376</td>
      <td>0.022726</td>
      <td>-0.017825</td>
      <td>0.044883</td>
      <td>0.010652</td>
      <td>-0.024778</td>
      <td>-0.015643</td>
      <td>-0.019100</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>0.097117</td>
      <td>-0.099260</td>
      <td>-0.029993</td>
      <td>-0.019908</td>
      <td>-0.050117</td>
      <td>1.000000</td>
      <td>0.128686</td>
      <td>0.059713</td>
      <td>-0.209294</td>
      <td>0.039936</td>
      <td>...</td>
      <td>0.076364</td>
      <td>-0.010691</td>
      <td>-0.022733</td>
      <td>-0.016921</td>
      <td>0.035715</td>
      <td>-0.087162</td>
      <td>-0.013098</td>
      <td>0.036178</td>
      <td>-0.038487</td>
      <td>0.049913</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.238363</td>
      <td>-0.152625</td>
      <td>-0.004511</td>
      <td>-0.111115</td>
      <td>0.611558</td>
      <td>0.128686</td>
      <td>1.000000</td>
      <td>-0.119617</td>
      <td>-0.380957</td>
      <td>0.101755</td>
      <td>...</td>
      <td>-0.016173</td>
      <td>0.088571</td>
      <td>-0.014176</td>
      <td>-0.004522</td>
      <td>-0.004522</td>
      <td>-0.014893</td>
      <td>0.061841</td>
      <td>0.015132</td>
      <td>-0.071029</td>
      <td>-0.018176</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.002471</td>
      <td>-0.029120</td>
      <td>0.029822</td>
      <td>0.045787</td>
      <td>0.060164</td>
      <td>0.059713</td>
      <td>-0.119617</td>
      <td>1.000000</td>
      <td>-0.083167</td>
      <td>0.038275</td>
      <td>...</td>
      <td>0.036786</td>
      <td>-0.008258</td>
      <td>-0.017560</td>
      <td>-0.013070</td>
      <td>0.030299</td>
      <td>-0.021547</td>
      <td>-0.010117</td>
      <td>0.012215</td>
      <td>0.014323</td>
      <td>0.091007</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>0.317987</td>
      <td>0.004469</td>
      <td>0.020764</td>
      <td>0.149744</td>
      <td>-0.495251</td>
      <td>-0.209294</td>
      <td>-0.380957</td>
      <td>-0.083167</td>
      <td>1.000000</td>
      <td>0.020060</td>
      <td>...</td>
      <td>-0.028685</td>
      <td>-0.012681</td>
      <td>-0.000835</td>
      <td>0.001853</td>
      <td>-0.033900</td>
      <td>0.249236</td>
      <td>-0.002593</td>
      <td>-0.198960</td>
      <td>0.035229</td>
      <td>-0.012639</td>
    </tr>
    <tr>
      <th>CentralAir</th>
      <td>0.146953</td>
      <td>-0.011803</td>
      <td>0.030692</td>
      <td>-0.021231</td>
      <td>0.166468</td>
      <td>0.039936</td>
      <td>0.101755</td>
      <td>0.038275</td>
      <td>0.020060</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.039299</td>
      <td>0.009771</td>
      <td>-0.085660</td>
      <td>0.015465</td>
      <td>-0.079604</td>
      <td>0.079661</td>
      <td>0.011971</td>
      <td>-0.037373</td>
      <td>0.069869</td>
      <td>0.006907</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>-0.065292</td>
      <td>0.061989</td>
      <td>-0.037305</td>
      <td>0.034737</td>
      <td>-0.102303</td>
      <td>0.036543</td>
      <td>-0.053614</td>
      <td>-0.011195</td>
      <td>-0.002538</td>
      <td>-0.156913</td>
      <td>...</td>
      <td>-0.018834</td>
      <td>-0.013308</td>
      <td>0.019394</td>
      <td>-0.021064</td>
      <td>0.042260</td>
      <td>-0.102871</td>
      <td>-0.016305</td>
      <td>0.051671</td>
      <td>0.023082</td>
      <td>-0.009407</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0.378195</td>
      <td>0.146402</td>
      <td>-0.006341</td>
      <td>0.082470</td>
      <td>0.291254</td>
      <td>0.058855</td>
      <td>0.165326</td>
      <td>0.013884</td>
      <td>-0.007214</td>
      <td>0.138629</td>
      <td>...</td>
      <td>0.031896</td>
      <td>0.038435</td>
      <td>-0.030948</td>
      <td>-0.009616</td>
      <td>0.000448</td>
      <td>0.024405</td>
      <td>-0.030815</td>
      <td>-0.007497</td>
      <td>-0.029901</td>
      <td>0.004695</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.386723</td>
      <td>0.436593</td>
      <td>0.029302</td>
      <td>0.333589</td>
      <td>0.067158</td>
      <td>-0.076072</td>
      <td>-0.027273</td>
      <td>-0.041978</td>
      <td>0.283160</td>
      <td>0.104650</td>
      <td>...</td>
      <td>0.014768</td>
      <td>-0.005304</td>
      <td>-0.008799</td>
      <td>0.001570</td>
      <td>-0.018359</td>
      <td>0.244249</td>
      <td>-0.019353</td>
      <td>-0.165823</td>
      <td>0.045597</td>
      <td>-0.026000</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.489782</td>
      <td>0.138347</td>
      <td>0.035087</td>
      <td>0.049242</td>
      <td>0.296970</td>
      <td>-0.018227</td>
      <td>0.140714</td>
      <td>-0.028726</td>
      <td>0.183303</td>
      <td>0.230741</td>
      <td>...</td>
      <td>-0.038068</td>
      <td>0.012220</td>
      <td>-0.002572</td>
      <td>-0.005535</td>
      <td>-0.041904</td>
      <td>0.296671</td>
      <td>-0.080601</td>
      <td>-0.218665</td>
      <td>-0.047794</td>
      <td>0.006372</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.456408</td>
      <td>0.190211</td>
      <td>0.019470</td>
      <td>0.102994</td>
      <td>0.220352</td>
      <td>-0.061106</td>
      <td>0.106398</td>
      <td>-0.022507</td>
      <td>0.250375</td>
      <td>0.170837</td>
      <td>...</td>
      <td>-0.039236</td>
      <td>0.004583</td>
      <td>-0.003825</td>
      <td>0.011800</td>
      <td>-0.038201</td>
      <td>0.327162</td>
      <td>-0.058890</td>
      <td>-0.231757</td>
      <td>-0.032854</td>
      <td>0.003240</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>0.166642</td>
      <td>0.064402</td>
      <td>0.029401</td>
      <td>-0.051575</td>
      <td>0.115843</td>
      <td>0.035070</td>
      <td>0.002365</td>
      <td>-0.004503</td>
      <td>0.042720</td>
      <td>0.265618</td>
      <td>...</td>
      <td>0.009940</td>
      <td>0.010801</td>
      <td>-0.056564</td>
      <td>-0.035304</td>
      <td>-0.038328</td>
      <td>0.070043</td>
      <td>-0.121200</td>
      <td>-0.036438</td>
      <td>0.032469</td>
      <td>0.005152</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.566024</td>
      <td>0.687501</td>
      <td>0.020643</td>
      <td>0.527567</td>
      <td>0.208171</td>
      <td>-0.009640</td>
      <td>0.044656</td>
      <td>-0.022963</td>
      <td>0.240257</td>
      <td>0.093666</td>
      <td>...</td>
      <td>0.030312</td>
      <td>0.008287</td>
      <td>-0.016628</td>
      <td>0.017268</td>
      <td>-0.022348</td>
      <td>0.168368</td>
      <td>-0.036522</td>
      <td>-0.121102</td>
      <td>0.044121</td>
      <td>-0.008545</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>-0.119916</td>
      <td>0.609707</td>
      <td>-0.004972</td>
      <td>0.210540</td>
      <td>0.004262</td>
      <td>-0.032148</td>
      <td>-0.007462</td>
      <td>0.001547</td>
      <td>-0.041118</td>
      <td>0.134637</td>
      <td>...</td>
      <td>-0.013854</td>
      <td>0.045466</td>
      <td>-0.059983</td>
      <td>0.001997</td>
      <td>-0.021325</td>
      <td>0.060505</td>
      <td>-0.034560</td>
      <td>-0.008467</td>
      <td>0.027628</td>
      <td>-0.019939</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0.068101</td>
      <td>0.059306</td>
      <td>-0.024600</td>
      <td>0.265391</td>
      <td>-0.081007</td>
      <td>-0.040751</td>
      <td>0.025907</td>
      <td>-0.058649</td>
      <td>0.030086</td>
      <td>-0.246797</td>
      <td>...</td>
      <td>-0.011083</td>
      <td>-0.007832</td>
      <td>0.023075</td>
      <td>0.040833</td>
      <td>-0.012396</td>
      <td>-0.041377</td>
      <td>0.059075</td>
      <td>0.009080</td>
      <td>0.013583</td>
      <td>-0.005536</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>0.299475</td>
      <td>0.050986</td>
      <td>0.020423</td>
      <td>0.113681</td>
      <td>0.214103</td>
      <td>0.111170</td>
      <td>0.227786</td>
      <td>0.038502</td>
      <td>-0.002618</td>
      <td>0.049755</td>
      <td>...</td>
      <td>-0.007818</td>
      <td>-0.002872</td>
      <td>-0.006018</td>
      <td>0.001076</td>
      <td>-0.015040</td>
      <td>0.020039</td>
      <td>-0.005722</td>
      <td>-0.002292</td>
      <td>-0.197131</td>
      <td>0.010123</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>0.245181</td>
      <td>0.042549</td>
      <td>0.023499</td>
      <td>0.121093</td>
      <td>0.076670</td>
      <td>-0.009312</td>
      <td>0.029931</td>
      <td>-0.028578</td>
      <td>0.160829</td>
      <td>-0.011683</td>
      <td>...</td>
      <td>0.014939</td>
      <td>-0.006010</td>
      <td>0.031665</td>
      <td>0.005374</td>
      <td>-0.011881</td>
      <td>0.183706</td>
      <td>0.001366</td>
      <td>-0.139867</td>
      <td>-0.025107</td>
      <td>-0.043535</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>-0.014241</td>
      <td>0.063353</td>
      <td>-0.004296</td>
      <td>0.149313</td>
      <td>-0.064503</td>
      <td>0.014807</td>
      <td>-0.042304</td>
      <td>-0.006376</td>
      <td>0.028167</td>
      <td>-0.050149</td>
      <td>...</td>
      <td>-0.006302</td>
      <td>-0.004453</td>
      <td>0.082887</td>
      <td>-0.007049</td>
      <td>-0.007049</td>
      <td>-0.036308</td>
      <td>-0.005456</td>
      <td>0.025586</td>
      <td>0.007724</td>
      <td>-0.003148</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>-0.251758</td>
      <td>0.307886</td>
      <td>-0.043825</td>
      <td>0.041622</td>
      <td>-0.069836</td>
      <td>-0.065649</td>
      <td>0.023578</td>
      <td>0.009469</td>
      <td>-0.140759</td>
      <td>-0.101774</td>
      <td>...</td>
      <td>0.028636</td>
      <td>0.028994</td>
      <td>0.085451</td>
      <td>-0.001244</td>
      <td>0.014005</td>
      <td>-0.045156</td>
      <td>-0.014555</td>
      <td>0.026359</td>
      <td>-0.024969</td>
      <td>-0.022844</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.339850</td>
      <td>0.173800</td>
      <td>0.019144</td>
      <td>0.093924</td>
      <td>0.261256</td>
      <td>-0.071330</td>
      <td>0.059590</td>
      <td>0.027440</td>
      <td>0.113862</td>
      <td>0.126409</td>
      <td>...</td>
      <td>-0.017731</td>
      <td>-0.021139</td>
      <td>0.001547</td>
      <td>0.015601</td>
      <td>-0.022686</td>
      <td>0.165692</td>
      <td>-0.025899</td>
      <td>-0.128187</td>
      <td>0.017108</td>
      <td>0.063452</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>-0.021096</td>
      <td>0.016197</td>
      <td>0.000354</td>
      <td>0.011634</td>
      <td>0.003571</td>
      <td>0.004940</td>
      <td>-0.016241</td>
      <td>-0.007392</td>
      <td>-0.023837</td>
      <td>-0.002478</td>
      <td>...</td>
      <td>-0.004596</td>
      <td>-0.003248</td>
      <td>0.002975</td>
      <td>0.013771</td>
      <td>-0.005140</td>
      <td>-0.026478</td>
      <td>-0.003979</td>
      <td>0.025009</td>
      <td>-0.022733</td>
      <td>-0.002296</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>0.031372</td>
      <td>0.035164</td>
      <td>0.029474</td>
      <td>0.046933</td>
      <td>-0.015727</td>
      <td>-0.015211</td>
      <td>-0.028738</td>
      <td>0.027816</td>
      <td>0.034888</td>
      <td>0.009846</td>
      <td>...</td>
      <td>0.003454</td>
      <td>-0.011263</td>
      <td>0.016522</td>
      <td>0.040735</td>
      <td>-0.054700</td>
      <td>0.094991</td>
      <td>0.028174</td>
      <td>-0.087446</td>
      <td>0.003690</td>
      <td>-0.051552</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0.211671</td>
      <td>0.208026</td>
      <td>-0.005842</td>
      <td>0.090935</td>
      <td>0.111761</td>
      <td>0.003093</td>
      <td>0.055175</td>
      <td>-0.028273</td>
      <td>0.129005</td>
      <td>0.025858</td>
      <td>...</td>
      <td>-0.021098</td>
      <td>-0.005122</td>
      <td>-0.030380</td>
      <td>0.023489</td>
      <td>-0.019525</td>
      <td>0.171467</td>
      <td>-0.025573</td>
      <td>-0.106605</td>
      <td>-0.005664</td>
      <td>0.028199</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>-0.144203</td>
      <td>0.028942</td>
      <td>0.025504</td>
      <td>0.003265</td>
      <td>-0.046231</td>
      <td>0.040229</td>
      <td>-0.054147</td>
      <td>0.098780</td>
      <td>-0.136841</td>
      <td>0.118969</td>
      <td>...</td>
      <td>0.031788</td>
      <td>-0.019156</td>
      <td>-0.064332</td>
      <td>0.001299</td>
      <td>-0.019779</td>
      <td>-0.156175</td>
      <td>-0.050663</td>
      <td>0.163684</td>
      <td>0.042848</td>
      <td>0.009994</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>0.476224</td>
      <td>0.295493</td>
      <td>0.030371</td>
      <td>0.082689</td>
      <td>0.239666</td>
      <td>-0.059119</td>
      <td>0.084653</td>
      <td>-0.032511</td>
      <td>0.308159</td>
      <td>0.272038</td>
      <td>...</td>
      <td>0.034147</td>
      <td>0.037524</td>
      <td>-0.037305</td>
      <td>0.004269</td>
      <td>-0.021172</td>
      <td>0.327412</td>
      <td>-0.057962</td>
      <td>-0.225013</td>
      <td>0.058823</td>
      <td>-0.001881</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.131525</td>
      <td>0.081487</td>
      <td>-0.007992</td>
      <td>0.072988</td>
      <td>0.140491</td>
      <td>0.041709</td>
      <td>0.076760</td>
      <td>0.016983</td>
      <td>-0.035092</td>
      <td>0.018122</td>
      <td>...</td>
      <td>-0.003600</td>
      <td>-0.002544</td>
      <td>-0.005410</td>
      <td>-0.004027</td>
      <td>-0.004027</td>
      <td>0.008838</td>
      <td>-0.003117</td>
      <td>0.002642</td>
      <td>0.004413</td>
      <td>-0.001798</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>PavedDrive_P</th>
      <td>-0.062613</td>
      <td>0.012140</td>
      <td>-0.016851</td>
      <td>0.010184</td>
      <td>-0.079791</td>
      <td>-0.041809</td>
      <td>-0.055867</td>
      <td>0.021285</td>
      <td>0.024459</td>
      <td>-0.079237</td>
      <td>...</td>
      <td>-0.007592</td>
      <td>-0.005364</td>
      <td>-0.011407</td>
      <td>-0.008491</td>
      <td>-0.008491</td>
      <td>-0.043737</td>
      <td>-0.006572</td>
      <td>0.056531</td>
      <td>0.009304</td>
      <td>-0.003792</td>
    </tr>
    <tr>
      <th>PavedDrive_Y</th>
      <td>0.163848</td>
      <td>-0.040082</td>
      <td>0.022902</td>
      <td>-0.054431</td>
      <td>0.191902</td>
      <td>0.068581</td>
      <td>0.092340</td>
      <td>0.029823</td>
      <td>-0.015493</td>
      <td>0.325482</td>
      <td>...</td>
      <td>0.015685</td>
      <td>0.011083</td>
      <td>-0.072014</td>
      <td>0.017543</td>
      <td>-0.025144</td>
      <td>0.081351</td>
      <td>-0.041492</td>
      <td>-0.065257</td>
      <td>0.019757</td>
      <td>0.007834</td>
    </tr>
    <tr>
      <th>PoolQC_Yes</th>
      <td>0.146727</td>
      <td>0.090073</td>
      <td>-0.008075</td>
      <td>0.074962</td>
      <td>0.166271</td>
      <td>0.043296</td>
      <td>0.089494</td>
      <td>0.021204</td>
      <td>-0.037279</td>
      <td>0.018311</td>
      <td>...</td>
      <td>-0.003638</td>
      <td>-0.002571</td>
      <td>-0.005466</td>
      <td>-0.004069</td>
      <td>-0.004069</td>
      <td>0.014872</td>
      <td>-0.003150</td>
      <td>-0.002186</td>
      <td>0.004459</td>
      <td>-0.001817</td>
    </tr>
    <tr>
      <th>RoofMatl_CompShg</th>
      <td>-0.190345</td>
      <td>-0.010648</td>
      <td>-0.007307</td>
      <td>-0.006294</td>
      <td>-0.105754</td>
      <td>-0.093091</td>
      <td>-0.114261</td>
      <td>-0.104081</td>
      <td>0.022127</td>
      <td>-0.014526</td>
      <td>...</td>
      <td>0.007058</td>
      <td>0.004987</td>
      <td>0.010605</td>
      <td>0.007893</td>
      <td>0.007893</td>
      <td>0.021945</td>
      <td>0.006110</td>
      <td>-0.021972</td>
      <td>-0.008650</td>
      <td>0.003525</td>
    </tr>
    <tr>
      <th>RoofMatl_Membran</th>
      <td>0.013574</td>
      <td>-0.020818</td>
      <td>-0.003046</td>
      <td>-0.025540</td>
      <td>-0.012497</td>
      <td>0.165014</td>
      <td>0.022216</td>
      <td>-0.005837</td>
      <td>-0.027930</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>0.010218</td>
      <td>0.001682</td>
      <td>-0.000685</td>
    </tr>
    <tr>
      <th>RoofMatl_Metal</th>
      <td>-0.011830</td>
      <td>-0.020818</td>
      <td>0.113083</td>
      <td>-0.041236</td>
      <td>0.028386</td>
      <td>-0.007557</td>
      <td>0.022216</td>
      <td>-0.005837</td>
      <td>-0.033620</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>0.010218</td>
      <td>0.001682</td>
      <td>-0.000685</td>
    </tr>
    <tr>
      <th>RoofMatl_Roll</th>
      <td>-0.015895</td>
      <td>0.038697</td>
      <td>-0.003046</td>
      <td>0.037244</td>
      <td>-0.012841</td>
      <td>-0.007557</td>
      <td>-0.018176</td>
      <td>-0.005837</td>
      <td>0.008343</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>0.010218</td>
      <td>0.001682</td>
      <td>-0.000685</td>
    </tr>
    <tr>
      <th>RoofMatl_Tar&amp;Grv</th>
      <td>0.071021</td>
      <td>-0.023777</td>
      <td>-0.010137</td>
      <td>0.009978</td>
      <td>0.015044</td>
      <td>0.088310</td>
      <td>0.037273</td>
      <td>0.156375</td>
      <td>-0.047352</td>
      <td>-0.009128</td>
      <td>...</td>
      <td>-0.004567</td>
      <td>-0.003227</td>
      <td>-0.006862</td>
      <td>-0.005108</td>
      <td>-0.005108</td>
      <td>-0.026310</td>
      <td>-0.003954</td>
      <td>0.034006</td>
      <td>0.005597</td>
      <td>-0.002281</td>
    </tr>
    <tr>
      <th>RoofMatl_WdShake</th>
      <td>0.096561</td>
      <td>0.009270</td>
      <td>-0.006820</td>
      <td>-0.005641</td>
      <td>0.005266</td>
      <td>0.012160</td>
      <td>0.049744</td>
      <td>-0.013070</td>
      <td>0.011726</td>
      <td>0.015465</td>
      <td>...</td>
      <td>-0.003073</td>
      <td>-0.002171</td>
      <td>-0.004617</td>
      <td>-0.003436</td>
      <td>-0.003436</td>
      <td>-0.017701</td>
      <td>-0.002660</td>
      <td>-0.011736</td>
      <td>0.003766</td>
      <td>-0.001535</td>
    </tr>
    <tr>
      <th>RoofMatl_WdShngl</th>
      <td>0.117333</td>
      <td>0.032092</td>
      <td>-0.007473</td>
      <td>0.016499</td>
      <td>0.070121</td>
      <td>0.003765</td>
      <td>0.071029</td>
      <td>0.025282</td>
      <td>0.031765</td>
      <td>0.016947</td>
      <td>...</td>
      <td>-0.003367</td>
      <td>-0.002379</td>
      <td>-0.005059</td>
      <td>-0.003766</td>
      <td>-0.003766</td>
      <td>-0.019397</td>
      <td>-0.002915</td>
      <td>0.025072</td>
      <td>0.004127</td>
      <td>-0.001682</td>
    </tr>
    <tr>
      <th>RoofStyle_Gable</th>
      <td>-0.314131</td>
      <td>0.086670</td>
      <td>-0.029938</td>
      <td>-0.013846</td>
      <td>-0.193130</td>
      <td>-0.070168</td>
      <td>-0.080428</td>
      <td>-0.029263</td>
      <td>-0.027912</td>
      <td>-0.005086</td>
      <td>...</td>
      <td>-0.003996</td>
      <td>0.019583</td>
      <td>-0.000711</td>
      <td>0.030996</td>
      <td>0.030996</td>
      <td>-0.055967</td>
      <td>0.023993</td>
      <td>0.038323</td>
      <td>0.017853</td>
      <td>0.013843</td>
    </tr>
    <tr>
      <th>RoofStyle_Gambrel</th>
      <td>-0.061752</td>
      <td>0.071027</td>
      <td>-0.010137</td>
      <td>0.038471</td>
      <td>-0.053247</td>
      <td>-0.017782</td>
      <td>-0.036051</td>
      <td>0.009874</td>
      <td>0.004991</td>
      <td>-0.073356</td>
      <td>...</td>
      <td>-0.004567</td>
      <td>-0.003227</td>
      <td>-0.006862</td>
      <td>-0.005108</td>
      <td>-0.005108</td>
      <td>-0.026310</td>
      <td>-0.003954</td>
      <td>0.034006</td>
      <td>0.005597</td>
      <td>-0.002281</td>
    </tr>
    <tr>
      <th>RoofStyle_Hip</th>
      <td>0.323994</td>
      <td>-0.113568</td>
      <td>0.030141</td>
      <td>0.000307</td>
      <td>0.213285</td>
      <td>0.033034</td>
      <td>0.078014</td>
      <td>-0.001522</td>
      <td>0.044537</td>
      <td>0.025256</td>
      <td>...</td>
      <td>0.007146</td>
      <td>-0.018280</td>
      <td>0.005225</td>
      <td>-0.028934</td>
      <td>-0.028934</td>
      <td>0.075468</td>
      <td>-0.022396</td>
      <td>-0.062128</td>
      <td>-0.022246</td>
      <td>-0.012922</td>
    </tr>
    <tr>
      <th>RoofStyle_Mansard</th>
      <td>0.000529</td>
      <td>0.073463</td>
      <td>-0.008075</td>
      <td>0.065054</td>
      <td>-0.048464</td>
      <td>0.008372</td>
      <td>-0.048189</td>
      <td>0.021204</td>
      <td>0.010849</td>
      <td>-0.021891</td>
      <td>...</td>
      <td>-0.003638</td>
      <td>-0.002571</td>
      <td>-0.005466</td>
      <td>-0.004069</td>
      <td>-0.004069</td>
      <td>-0.020959</td>
      <td>-0.003150</td>
      <td>0.027090</td>
      <td>0.004459</td>
      <td>-0.001817</td>
    </tr>
    <tr>
      <th>RoofStyle_Shed</th>
      <td>0.017622</td>
      <td>0.032125</td>
      <td>-0.004309</td>
      <td>-0.006525</td>
      <td>0.035284</td>
      <td>0.013539</td>
      <td>0.088571</td>
      <td>-0.008258</td>
      <td>-0.017125</td>
      <td>0.009771</td>
      <td>...</td>
      <td>-0.001941</td>
      <td>-0.001372</td>
      <td>-0.002917</td>
      <td>-0.002171</td>
      <td>-0.002171</td>
      <td>-0.011184</td>
      <td>-0.001681</td>
      <td>0.014455</td>
      <td>0.002379</td>
      <td>-0.000970</td>
    </tr>
    <tr>
      <th>SaleCondition_AdjLand</th>
      <td>-0.037451</td>
      <td>-0.014533</td>
      <td>-0.006098</td>
      <td>0.048378</td>
      <td>-0.014874</td>
      <td>-0.015130</td>
      <td>-0.016173</td>
      <td>0.182202</td>
      <td>-0.034618</td>
      <td>-0.092426</td>
      <td>...</td>
      <td>-0.002747</td>
      <td>-0.001941</td>
      <td>-0.004128</td>
      <td>-0.003073</td>
      <td>-0.003073</td>
      <td>-0.015827</td>
      <td>-0.002378</td>
      <td>0.020457</td>
      <td>0.003367</td>
      <td>-0.001372</td>
    </tr>
    <tr>
      <th>SaleCondition_Alloca</th>
      <td>0.068107</td>
      <td>-0.020234</td>
      <td>-0.010591</td>
      <td>0.052186</td>
      <td>0.021369</td>
      <td>-0.026277</td>
      <td>0.194292</td>
      <td>-0.020297</td>
      <td>-0.059130</td>
      <td>-0.006741</td>
      <td>...</td>
      <td>-0.004772</td>
      <td>-0.003372</td>
      <td>-0.007170</td>
      <td>-0.005337</td>
      <td>-0.005337</td>
      <td>-0.027489</td>
      <td>-0.004131</td>
      <td>0.035530</td>
      <td>-0.112734</td>
      <td>-0.002383</td>
    </tr>
    <tr>
      <th>SaleCondition_Family</th>
      <td>0.021949</td>
      <td>-0.027180</td>
      <td>-0.013711</td>
      <td>0.066381</td>
      <td>0.000765</td>
      <td>-0.007929</td>
      <td>-0.018183</td>
      <td>0.039116</td>
      <td>0.021534</td>
      <td>-0.016691</td>
      <td>...</td>
      <td>0.106555</td>
      <td>-0.004365</td>
      <td>-0.009282</td>
      <td>-0.006909</td>
      <td>-0.006909</td>
      <td>-0.035587</td>
      <td>-0.005348</td>
      <td>0.028599</td>
      <td>0.007571</td>
      <td>-0.003085</td>
    </tr>
    <tr>
      <th>SaleCondition_Normal</th>
      <td>-0.158772</td>
      <td>0.031766</td>
      <td>-0.009177</td>
      <td>0.003551</td>
      <td>-0.019560</td>
      <td>0.041207</td>
      <td>-0.022306</td>
      <td>-0.034388</td>
      <td>-0.153930</td>
      <td>-0.014821</td>
      <td>...</td>
      <td>-0.043784</td>
      <td>0.017320</td>
      <td>-0.031583</td>
      <td>-0.003139</td>
      <td>0.027414</td>
      <td>-0.645698</td>
      <td>-0.097031</td>
      <td>0.634322</td>
      <td>-0.002140</td>
      <td>-0.055982</td>
    </tr>
    <tr>
      <th>SaleCondition_Partial</th>
      <td>0.221037</td>
      <td>0.004852</td>
      <td>0.018526</td>
      <td>-0.060265</td>
      <td>0.044912</td>
      <td>-0.085761</td>
      <td>-0.004721</td>
      <td>-0.022949</td>
      <td>0.249315</td>
      <td>0.080725</td>
      <td>...</td>
      <td>-0.016038</td>
      <td>-0.011333</td>
      <td>0.007176</td>
      <td>-0.017938</td>
      <td>-0.017938</td>
      <td>0.986819</td>
      <td>-0.013885</td>
      <td>-0.769559</td>
      <td>0.019657</td>
      <td>-0.008011</td>
    </tr>
    <tr>
      <th>SaleType_CWD</th>
      <td>0.033381</td>
      <td>0.007628</td>
      <td>-0.006098</td>
      <td>0.030047</td>
      <td>0.008951</td>
      <td>0.076364</td>
      <td>-0.016173</td>
      <td>0.036786</td>
      <td>-0.028685</td>
      <td>-0.039299</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.001941</td>
      <td>-0.004128</td>
      <td>-0.003073</td>
      <td>-0.003073</td>
      <td>-0.015827</td>
      <td>-0.002378</td>
      <td>-0.134295</td>
      <td>0.003367</td>
      <td>-0.001372</td>
    </tr>
    <tr>
      <th>SaleType_Con</th>
      <td>0.007559</td>
      <td>0.003778</td>
      <td>-0.004309</td>
      <td>-0.028730</td>
      <td>0.030694</td>
      <td>-0.010691</td>
      <td>0.088571</td>
      <td>-0.008258</td>
      <td>-0.012681</td>
      <td>0.009771</td>
      <td>...</td>
      <td>-0.001941</td>
      <td>1.000000</td>
      <td>-0.002917</td>
      <td>-0.002171</td>
      <td>-0.002171</td>
      <td>-0.011184</td>
      <td>-0.001681</td>
      <td>-0.094896</td>
      <td>0.002379</td>
      <td>-0.000970</td>
    </tr>
    <tr>
      <th>SaleType_ConLD</th>
      <td>-0.011789</td>
      <td>-0.018808</td>
      <td>-0.009162</td>
      <td>-0.013874</td>
      <td>-0.021376</td>
      <td>-0.022733</td>
      <td>-0.014176</td>
      <td>-0.017560</td>
      <td>-0.000835</td>
      <td>-0.085660</td>
      <td>...</td>
      <td>-0.004128</td>
      <td>-0.002917</td>
      <td>1.000000</td>
      <td>-0.004617</td>
      <td>-0.004617</td>
      <td>-0.023782</td>
      <td>-0.003574</td>
      <td>-0.201789</td>
      <td>-0.131726</td>
      <td>-0.002062</td>
    </tr>
    <tr>
      <th>SaleType_ConLI</th>
      <td>0.006094</td>
      <td>0.016175</td>
      <td>-0.006820</td>
      <td>0.081051</td>
      <td>0.022726</td>
      <td>-0.016921</td>
      <td>-0.004522</td>
      <td>-0.013070</td>
      <td>0.001853</td>
      <td>0.015465</td>
      <td>...</td>
      <td>-0.003073</td>
      <td>-0.002171</td>
      <td>-0.004617</td>
      <td>1.000000</td>
      <td>-0.003436</td>
      <td>-0.017701</td>
      <td>-0.002660</td>
      <td>-0.150198</td>
      <td>0.003766</td>
      <td>-0.001535</td>
    </tr>
    <tr>
      <th>SaleType_ConLw</th>
      <td>-0.043721</td>
      <td>0.012602</td>
      <td>-0.006820</td>
      <td>-0.029071</td>
      <td>-0.017825</td>
      <td>0.035715</td>
      <td>-0.004522</td>
      <td>0.030299</td>
      <td>-0.033900</td>
      <td>-0.079604</td>
      <td>...</td>
      <td>-0.003073</td>
      <td>-0.002171</td>
      <td>-0.004617</td>
      <td>-0.003436</td>
      <td>1.000000</td>
      <td>-0.017701</td>
      <td>-0.002660</td>
      <td>-0.150198</td>
      <td>0.003766</td>
      <td>-0.001535</td>
    </tr>
    <tr>
      <th>SaleType_New</th>
      <td>0.221219</td>
      <td>0.010810</td>
      <td>0.019596</td>
      <td>-0.054679</td>
      <td>0.044883</td>
      <td>-0.087162</td>
      <td>-0.014893</td>
      <td>-0.021547</td>
      <td>0.249236</td>
      <td>0.079661</td>
      <td>...</td>
      <td>-0.015827</td>
      <td>-0.011184</td>
      <td>-0.023782</td>
      <td>-0.017701</td>
      <td>-0.017701</td>
      <td>1.000000</td>
      <td>-0.013702</td>
      <td>-0.773680</td>
      <td>0.019397</td>
      <td>-0.007905</td>
    </tr>
    <tr>
      <th>SaleType_Oth</th>
      <td>-0.008215</td>
      <td>-0.036082</td>
      <td>-0.005279</td>
      <td>0.022234</td>
      <td>0.010652</td>
      <td>-0.013098</td>
      <td>0.061841</td>
      <td>-0.010117</td>
      <td>-0.002593</td>
      <td>0.011971</td>
      <td>...</td>
      <td>-0.002378</td>
      <td>-0.001681</td>
      <td>-0.003574</td>
      <td>-0.002660</td>
      <td>-0.002660</td>
      <td>-0.013702</td>
      <td>1.000000</td>
      <td>-0.116263</td>
      <td>0.002915</td>
      <td>-0.001188</td>
    </tr>
    <tr>
      <th>SaleType_WD</th>
      <td>-0.198056</td>
      <td>0.026769</td>
      <td>-0.014211</td>
      <td>0.046933</td>
      <td>-0.024778</td>
      <td>0.036178</td>
      <td>0.015132</td>
      <td>0.012215</td>
      <td>-0.198960</td>
      <td>-0.037373</td>
      <td>...</td>
      <td>-0.134295</td>
      <td>-0.094896</td>
      <td>-0.201789</td>
      <td>-0.150198</td>
      <td>-0.150198</td>
      <td>-0.773680</td>
      <td>-0.116263</td>
      <td>1.000000</td>
      <td>0.006539</td>
      <td>-0.067078</td>
    </tr>
    <tr>
      <th>Street_Pave</th>
      <td>0.005950</td>
      <td>0.046983</td>
      <td>0.007473</td>
      <td>0.030573</td>
      <td>-0.015643</td>
      <td>-0.038487</td>
      <td>-0.071029</td>
      <td>0.014323</td>
      <td>0.035229</td>
      <td>0.069869</td>
      <td>...</td>
      <td>0.003367</td>
      <td>0.002379</td>
      <td>-0.131726</td>
      <td>0.003766</td>
      <td>0.003766</td>
      <td>0.019397</td>
      <td>0.002915</td>
      <td>0.006539</td>
      <td>1.000000</td>
      <td>0.001682</td>
    </tr>
    <tr>
      <th>Utilities_NoSeWa</th>
      <td>0.012287</td>
      <td>-0.020818</td>
      <td>-0.003046</td>
      <td>0.000620</td>
      <td>-0.019100</td>
      <td>0.049913</td>
      <td>-0.018176</td>
      <td>0.091007</td>
      <td>-0.012639</td>
      <td>0.006907</td>
      <td>...</td>
      <td>-0.001372</td>
      <td>-0.000970</td>
      <td>-0.002062</td>
      <td>-0.001535</td>
      <td>-0.001535</td>
      <td>-0.007905</td>
      <td>-0.001188</td>
      <td>-0.067078</td>
      <td>0.001682</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>242 rows × 242 columns</p>
</div>



**This is the start of setting the variables into x and y. For x, we use all the variables except for the column "SalePrice:. "SalePrice" is our y and the variable we want to predict in the test data. **


```python
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib
train = train.fillna(0)
x = train.drop('SalePrice', 1)
y = train[['SalePrice']]
x.shape, y.shape
```

    /Users/YannellySanchez/anaconda/lib/python2.7/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


    Using matplotlib backend: MacOSX





    ((1460, 241), (1460, 1))



**In order to train the dataset wemust split it to the  train and test. By training this dataset it allows for the prediction of the test dataset.**


```python
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5, random_state=10)
```

    /Users/YannellySanchez/anaconda/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
x_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 730 entries, 854 to 294
    Columns: 241 entries, 1stFlrSF to Utilities_NoSeWa
    dtypes: float64(11), int64(230)
    memory usage: 1.3 MB


### Mulitple algorithms
I tested mulitple algorthms to determine the best one for the prediction. 


```python
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
clf = clf.fit(x_train, y_train)
clf
y_predictgrad = clf.predict(x_test)

```

    /Users/YannellySanchez/anaconda/lib/python2.7/site-packages/sklearn/utils/validation.py:526: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)



```python
clf.score(x_test, y_test)
```




    0.88095575779064661




```python
mean_squared_error(y_test, y_predictgrad)**0.5
```




    28038.6258723459




```python
#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr =rfr.fit(x_train, y_train)
rfr
y_pred_randomforest = rfr.predict(x_test)

```

    /Users/YannellySanchez/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().



```python
rfr.score(x_test, y_test)
```




    0.84448231320290656




```python
mean_squared_error(y_test, y_pred_randomforest)**0.5
```




    32047.372048802536




```python
#linear regression
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
reg = linear_model.LinearRegression()
regmodel = reg.fit(x_train, y_train)
y_predict = regmodel.predict(x_test)
regmodel.score(x_test, y_test)
```




    -568456850.86376715




```python
RMSEtrain = mean_squared_error(y_train, y_predict)**0.5
RMSEtest = mean_squared_error(y_test, y_predict)**0.5
print ('RMSE for the training data is: ', RMSEtrain)
print ('RMSE for the testing data is: ', RMSEtest)
```

    ('RMSE for the training data is: ', 1937545677.1660187)
    ('RMSE for the testing data is: ', 1937542694.5464532)


**I included Ridge and Lasso Regression so see the score, but this project is meant for a more linear algorithm.** 


```python
# Ridge Regression model
ridgereg = linear_model.Ridge(alpha =0.1)
ridgereg.fit(x_train, y_train)
y_pred_ridge = ridgereg.predict(x_test)
ridgereg.score(x_test, y_test)
```




    0.79915526382634638




```python
mean_squared_error(y_test, y_pred_ridge)**0.5
```




    36419.397377775676




```python
#Lasso Regression model
lassoreg = linear_model.Lasso(alpha=0.1)
lassoreg.fit(x_train, y_train)
y_pred_lasso = lassoreg.predict(x_test)
lassoreg.score(x_test, y_test)
```

    /Users/YannellySanchez/anaconda/lib/python2.7/site-packages/sklearn/linear_model/coordinate_descent.py:484: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)





    0.47237620229488153




```python
mean_squared_error(y_test, y_pred_lasso)**0.5
```




    59028.903515732716




```python
from sklearn.tree import DecisionTreeRegressor
Decision = DecisionTreeRegressor()
Decision.fit(x_train, y_train)
y_DecisionTree = Decision.predict(x_train)
Decision.score(x_train, y_test)

```




    -1.0321587500602374




```python
mean_squared_error(y_test, y_DecisionTree)**0.5
```




    115846.08037443714




```python
x_train.shape, x_test.shape,y_train.shape, y_test.shape
```




    ((730, 241), (730, 241), (730, 1), (730, 1))




```python
#Logistic Regression
from sklearn.linear_model import LogisticRegression as LogR
logistic = LogR()
logistic.fit(x_train, y_train)
log = logistic.predict(x_train)
logistic.score(x_train, y_train)

```




    0.95753424657534247




```python
mean_squared_error(y_test, log)**0.5
```




    115736.53095887451



**I did at one point I tried stacking(where we used muliple algorithm to find a better prediction. You add each prediction into into own column and then use an agorithm to find the final prediction.), but I got a better prediction with a simple Logistic Regression.**


```python
test = pd.read_csv('test_data.csv')
test.info()
test = test.fillna(0)
test =test.drop('Unnamed: 0', 1)
testfinal = test.drop('Id', 1)
testfinal = testfinal.drop('SalePrice',1)
testfinal.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Columns: 244 entries, Unnamed: 0 to Utilities_NoSeWa
    dtypes: float64(12), int64(232)
    memory usage: 2.7 MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Columns: 241 entries, 1stFlrSF to Utilities_NoSeWa
    dtypes: float64(11), int64(230)
    memory usage: 2.7 MB


**testfinal and x_train should be the same shape, but I always double check. You just never know.**


```python
testfinal.shape, x_train.shape
```




    ((1459, 241), (730, 241))




```python
testfinal.head()
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
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>468.0</td>
      <td>144.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>270.0</td>
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
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>923.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>406.0</td>
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
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>9</td>
      <td>791.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137.0</td>
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
      <td>926</td>
      <td>678</td>
      <td>0</td>
      <td>9</td>
      <td>602.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>324.0</td>
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
      <td>1280</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1017.0</td>
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
<p>5 rows × 241 columns</p>
</div>



**I assign the variable "final" to the testfinal prediction.**


```python
final = logistic.predict(testfinal)
```

**I assigned a new variable to only the column 'Id" from the test dataset.**


```python
finaloutput = test[['Id']]
finaloutput.shape
```




    (1459, 1)



**I add a column to the new variable (dataframe) "finaloutput" called "SalePrice".**


```python
finaloutput['SalePrice']=final
finaloutput.head()
```

    /Users/YannellySanchez/anaconda/lib/python2.7/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':





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
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>129000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>128950.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>175000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>215000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>245500.0</td>
    </tr>
  </tbody>
</table>
</div>



**I save the dataframe in a .csv file while removing the index by making it False.


```python
finaloutput.to_csv('/Users/YannellySanchez/Google Drive/finaloutput.csv', index=False)
```
