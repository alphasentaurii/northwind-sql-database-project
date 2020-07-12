# Northwind SQL Database Project

![GitHub repo size](https://img.shields.io/github/repo-size/hakkeray/northwind-sql-database-project)
![GitHub license](https://img.shields.io/github/license/hakkeray/northwind-sql-database-project?color=black)

The `Northwind SQL Database Project` demonstrates how to use `SQL queries` and `hypothesis testing` in order to `recommend business strategies` for increasing sales and reducing costs for the fictitious "Northwind" company. The Northwind SQL database was created by Microsoft for data scientists to practice SQL queries and hypothesis testing in their analyses.

![northwind](https://github.com/hakkeray/dsc-mod-3-project-online-ds-ft-100719/blob/master/Northwind_ERD_updated.png?raw=true)

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed the latest version of `Jupyter Notebook`
* You have a `<Windows/Linux/Mac>` machine.

### FSDS - standard data science tools

You can do a quick install of all the standard data science libraries (matplotlib, pandas, etc) using the FSDS pypi package from `James Irving`:

```bash
pip install -U fsds_100719
```

## Running the Northwind SQL Database Project

To run this project locally, follow these steps:

In the command line/terminal:

```bash
git clone https://github.com/hakkeray/northwind-sql-database-project
cd northwind-sql-database-project
jupyter notebook
```

## Hypothesis Testing

Below are 4 hypotheses (each including a null hypothesis and alternative hypothesis) which I will test for statistical significance to determine if there are any relationships which would be useful from a strategic business perspective. Following this I will summarize the results, make final recommendations, and propose ideas for future analytical work.

---

## Objectives

**H1: Discount and Order Quantity**

Does discount amount have a statistically significant effect on order quantity? If so, at what level(s) of discount?

**H2: Countries and Order Quantity: Discount vs Full Price**

Do order quantities of individual countries differ when discounted vs full price?

**H3: Region and Order Revenue**

Does region have a statistically significant effect on average revenue per order?

**H4: Month and Order Quantity**

Does time of year have a statistically significant effect on average revenue per order?

## Process Outline

Outline of process I will follow in order to answer questions above:

-Question
1. Hypotheses
2. Exploratory Data Analysis (EDA)

-Select dataset
-Group data
-Explore data

3. Assumption Tests:
-Sample size
-Normality and Variance

4. Statistical Tests:
-Statistical test
-Effect size (if necessary)
-Post-hoc tests (if necessary)

5. Summarize Results

---

## Statistical Analysis Pipeline

For #3 and #4 above (Assumption and Statistical Tests):

  1. Check if sample sizes allow us to ignore assumptions by visualizing sample size comparisons for two groups (normality check).
      * Bar Plot: SEM (Standard Error of the Mean)

  2. If above test fails, check for normality and homogeneity of variance:
      * Test Assumption Normality:
          - D'Agostino-Pearson: scipy.stats.normaltest
          - Shapiro-Wilik Test: scipy.stats.shapiro
      
      * Test for Homogeneity of Variance:
          - Levene's Test: scipy.stats.levene)
  Parametric tests (means)	Nonparametric tests (medians)
  1-sample t test	1-sample Sign, 1-sample Wilcoxon
  2-sample t test	Mann-Whitney tes
  One-Way ANOVA	Kruskal-Wallis, Mood’s median tes
  Factorial DOE with one factor and one blocking variable	Friedman test

  3. Choose appropriate test based on above
      * T Test (1-sample)
          - `stats.ttest_1samp()`
      * T Test (2-sample)
          - stats.ttest_ind()
      * Welch's T-Test (2-sample)
          - stats.ttest_ind(equal_var=False)
      * Mann Whitney U
          - stats.mannwhitneyu()
      * ANOVA
          - stats.f_oneway()

  4. Calculate effect size for significant results.
      * Effect size: 
          - cohen's d

      -Interpretation:
      - Small effect = 0.2 ( cannot be seen by naked eye)
      - Medium effect = 0.5
      - Large Effect = 0.8 (can be seen by naked eye)


  5. If significant, follow up with post-hoc tests (if have more than 2 groups)
      * Tukey's
          - statsmodels.stats.multicomp.pairwise_tukeyhsd

## Contact

If you want to contact me you can reach me at <rukeine@gmail.com>.

## License

This project uses the following license: [MIT License](./LICENSE.md).

```python
#         _ __ _   _
#  /\_/\ | '__| | | |
#  [===] | |  | |_| |
#   \./  |_|   \__,_|
```
---

## Project Demo

```python
# connect to database / import data
import sqlite3
conn = sqlite3.connect('Northwind_small.sqlite')
cur = conn.cursor()
```

```python
# function for converting tables into dataframes on the fly
def get_table(cur, table):
    cur.execute(f"SELECT * from {table};")
    df = pd.DataFrame(cur.fetchall())
    df.columns = [desc[0] for desc in cur.description]
    return df
```

```python
# create dataframe of table names for referencing purposes
cur.execute("""SELECT name from sqlite_master WHERE type='table';""")
df_tables = pd.DataFrame(cur.fetchall(), columns=['Table'])
df_tables
```

<html>
<body>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Employee</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Category</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Customer</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Shipper</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Supplier</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Order</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Product</td>
    </tr>
    <tr>
      <td>7</td>
      <td>OrderDetail</td>
    </tr>
    <tr>
      <td>8</td>
      <td>CustomerCustomerDemo</td>
    </tr>
    <tr>
      <td>9</td>
      <td>CustomerDemographic</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Region</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Territory</td>
    </tr>
    <tr>
      <td>12</td>
      <td>EmployeeTerritory</td>
    </tr>
  </tbody>
</table>
</div>
</body>
</html>

# H1: Discount--Quantity

* Does discount amount have a statistically significant effect on the quantity of a product in an order? 
* If so, at what level(s) of discount?

## Hypotheses
- $H_0$: Discount amount has no relationship with the quantity of a product in an order.
- $H_A$: Discount amount has a statistically significant effect on the quantity in an order.

- $\alpha$=0.05

## EDA
Select the proper dataset for analysis, perform EDA, and generate data groups for testing.

### Select dataset


```python
df_orderDetail = get_table(cur, 'OrderDetail')
df_orderDetail.head()
```
<html>
<body>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
</body>
</html>

### Group

```python
# check value counts for each level of discount
df_orderDetail['Discount'].value_counts()
```




    0.00    1317
    0.05     185
    0.10     173
    0.20     161
    0.15     157
    0.25     154
    0.03       3
    0.02       2
    0.01       1
    0.04       1
    0.06       1
    Name: Discount, dtype: int64




```python
# insert boolean column showing whether or not an order was discounted
df_orderDetail['discounted'] = np.where(df_orderDetail['Discount'] == 0.0, 0, 1)

# compare number of discount vs fullprice orders
df_orderDetail['discounted'].value_counts()
```




    0    1317
    1     838
    Name: discounted, dtype: int64




```python
# split orders into two groups (series): discount and fullprice order quantity
fullprice = df_orderDetail.groupby('discounted').get_group(0)['Quantity']
discount = df_orderDetail.groupby('discounted').get_group(1)['Quantity']
```

### Explore


```python
diff = (discount.mean() - fullprice.mean())
diff
```




    5.394523243866239




```python
# visually inspect differences in mean and StDev of distributions
sns.set_style("whitegrid")
%config InlineBackend.figure_format='retina'
%matplotlib inline
fig = plt.figure(figsize=(10,8))
ax = fig.gca()

ax.axvline(fullprice.mean(), color='blue', lw=2, ls='--', label='FP Avg')
ax.axvline(discount.mean(), color='orange', lw=2, ls='--', label='DC Avg')

fdict = {'fontfamily': 'PT Mono','fontsize': 16}

sns.distplot(fullprice, ax=ax, hist=True, kde=True, color='blue')
sns.distplot(discount, ax=ax, hist=True, kde=True, color='orange')
ax.legend(['Full Price', 'Discount'])
ax.set_title("Distribution of Full Price vs Discount Order Quantity", fontdict=fdict)
```




    Text(0.5, 1.0, 'Distribution of Full Price vs Discount Order Quantity')




![png](/output/output_25_1.png)



```python
fig = plt.figure(figsize=(10,8))
ax = fig.gca()
ax = sns.barplot(x='Discount', y='Quantity', data=df_orderDetail)
ax.set_title('Discount Levels and Order Qty', fontdict={'family': 'PT Mono', 'size':16})
```




    Text(0.5, 1.0, 'Discount Levels and Order Qty')




![png](/output/output_26_1.png)


We can already see that there is a clear relationship between order quantity and specific discount levels before running any statistical tests. However, what is more interesting to note from the visualization above is that the discount levels that DO have an effect appear to be very similar as far as the mean order quantity. The indication is that discount amount produces diminishing returns (offering a discount higher than 5% - the minimum effective amount - does not actually produce higher order quantity which means we are losing revenue we would have otherwise captured).

## Assumption Tests
**Select the appropriate t-test based on tests for the assumptions of normality and homogeneity of variance.**

### Sample Size
Check if sample sizes allow us to ignore assumptions; if not, test assumption normality.


```python
# visualize sample size comparisons for two groups (normality check)
import scipy.stats as stat
plt.bar(x='Full Price', height=fullprice.mean(), yerr=stat.sem(fullprice))
plt.bar(x='Discount', height=discount.mean(), yerr=stat.sem(discount))
plt.title("Order Quantity Sample Sizes: Full Price vs Discount")
```




    Text(0.5, 1.0, 'Order Quantity Sample Sizes: Full Price vs Discount')




![png](/output/output_30_1.png)


### Normality Test
Check assumptions of normality and homogeneity of variance


```python
# Test for normality - D'Agostino-Pearson's normality test: scipy.stats.normaltest
stat.normaltest(fullprice), stat.normaltest(discount)
```




    (NormaltestResult(statistic=544.5770045551502, pvalue=5.579637380545965e-119),
     NormaltestResult(statistic=261.528012299789, pvalue=1.6214878452829618e-57))



Failed normality test (p-values < 0.05). Run non-parametric test:


```python
# Run non-parametric test (since normality test failed)
stat.mannwhitneyu(fullprice, discount)
```




    MannwhitneyuResult(statistic=461541.0, pvalue=6.629381826999866e-11)



### Statistical Test
Perform chosen statistical test.


```python
# run tukey test for OQD (Order Quantity Discount) 
data = df_orderDetail['Quantity'].values
labels = df_orderDetail['Discount'].values

import statsmodels.api as sms
model = sms.stats.multicomp.pairwise_tukeyhsd(data,labels)
```


```python
# save OQD tukey test model results into dataframe (OQD: order quantity discount)
tukey_OQD = pd.DataFrame(data=model._results_table[1:], columns=model._results_table[0])
tukey_OQD
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>-19.7153</td>
      <td>0.9</td>
      <td>-80.3306</td>
      <td>40.9001</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>-19.7153</td>
      <td>0.9</td>
      <td>-62.593</td>
      <td>23.1625</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>-20.0486</td>
      <td>0.725</td>
      <td>-55.0714</td>
      <td>14.9742</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>-20.7153</td>
      <td>0.9</td>
      <td>-81.3306</td>
      <td>39.9001</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>6.2955</td>
      <td>0.0011</td>
      <td>1.5381</td>
      <td>11.053</td>
      <td>True</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>-19.7153</td>
      <td>0.9</td>
      <td>-80.3306</td>
      <td>40.9001</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>3.5217</td>
      <td>0.4269</td>
      <td>-1.3783</td>
      <td>8.4217</td>
      <td>False</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.0</td>
      <td>0.15</td>
      <td>6.6669</td>
      <td>0.0014</td>
      <td>1.551</td>
      <td>11.7828</td>
      <td>True</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>5.3096</td>
      <td>0.0303</td>
      <td>0.2508</td>
      <td>10.3684</td>
      <td>True</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>6.525</td>
      <td>0.0023</td>
      <td>1.3647</td>
      <td>11.6852</td>
      <td>True</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>-74.2101</td>
      <td>74.2101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>-0.3333</td>
      <td>0.9</td>
      <td>-70.2993</td>
      <td>69.6326</td>
      <td>False</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>-1.0</td>
      <td>0.9</td>
      <td>-86.6905</td>
      <td>84.6905</td>
      <td>False</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>26.0108</td>
      <td>0.9</td>
      <td>-34.745</td>
      <td>86.7667</td>
      <td>False</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.01</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>-85.6905</td>
      <td>85.6905</td>
      <td>False</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>23.237</td>
      <td>0.9</td>
      <td>-37.5302</td>
      <td>84.0042</td>
      <td>False</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.01</td>
      <td>0.15</td>
      <td>26.3822</td>
      <td>0.9</td>
      <td>-34.4028</td>
      <td>87.1671</td>
      <td>False</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.01</td>
      <td>0.2</td>
      <td>25.0248</td>
      <td>0.9</td>
      <td>-35.7554</td>
      <td>85.805</td>
      <td>False</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.01</td>
      <td>0.25</td>
      <td>26.2403</td>
      <td>0.9</td>
      <td>-34.5485</td>
      <td>87.029</td>
      <td>False</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>-0.3333</td>
      <td>0.9</td>
      <td>-55.6463</td>
      <td>54.9796</td>
      <td>False</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>-1.0</td>
      <td>0.9</td>
      <td>-75.2101</td>
      <td>73.2101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>26.0108</td>
      <td>0.6622</td>
      <td>-17.0654</td>
      <td>69.087</td>
      <td>False</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.02</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>-74.2101</td>
      <td>74.2101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.02</td>
      <td>0.1</td>
      <td>23.237</td>
      <td>0.7914</td>
      <td>-19.8552</td>
      <td>66.3292</td>
      <td>False</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.02</td>
      <td>0.15</td>
      <td>26.3822</td>
      <td>0.6461</td>
      <td>-16.7351</td>
      <td>69.4994</td>
      <td>False</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.02</td>
      <td>0.2</td>
      <td>25.0248</td>
      <td>0.7089</td>
      <td>-18.0857</td>
      <td>68.1354</td>
      <td>False</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.02</td>
      <td>0.25</td>
      <td>26.2403</td>
      <td>0.6528</td>
      <td>-16.8823</td>
      <td>69.3628</td>
      <td>False</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>-0.6667</td>
      <td>0.9</td>
      <td>-70.6326</td>
      <td>69.2993</td>
      <td>False</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>26.3441</td>
      <td>0.3639</td>
      <td>-8.9214</td>
      <td>61.6096</td>
      <td>False</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.03</td>
      <td>0.06</td>
      <td>0.3333</td>
      <td>0.9</td>
      <td>-69.6326</td>
      <td>70.2993</td>
      <td>False</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.03</td>
      <td>0.1</td>
      <td>23.5703</td>
      <td>0.5338</td>
      <td>-11.7147</td>
      <td>58.8553</td>
      <td>False</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.03</td>
      <td>0.15</td>
      <td>26.7155</td>
      <td>0.3436</td>
      <td>-8.6001</td>
      <td>62.0311</td>
      <td>False</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.03</td>
      <td>0.2</td>
      <td>25.3582</td>
      <td>0.428</td>
      <td>-9.9492</td>
      <td>60.6656</td>
      <td>False</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.03</td>
      <td>0.25</td>
      <td>26.5736</td>
      <td>0.3525</td>
      <td>-8.7485</td>
      <td>61.8957</td>
      <td>False</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>27.0108</td>
      <td>0.9</td>
      <td>-33.745</td>
      <td>87.7667</td>
      <td>False</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.04</td>
      <td>0.06</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>-84.6905</td>
      <td>86.6905</td>
      <td>False</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.04</td>
      <td>0.1</td>
      <td>24.237</td>
      <td>0.9</td>
      <td>-36.5302</td>
      <td>85.0042</td>
      <td>False</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.04</td>
      <td>0.15</td>
      <td>27.3822</td>
      <td>0.9</td>
      <td>-33.4028</td>
      <td>88.1671</td>
      <td>False</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.04</td>
      <td>0.2</td>
      <td>26.0248</td>
      <td>0.9</td>
      <td>-34.7554</td>
      <td>86.805</td>
      <td>False</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.04</td>
      <td>0.25</td>
      <td>27.2403</td>
      <td>0.9</td>
      <td>-33.5485</td>
      <td>88.029</td>
      <td>False</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>-26.0108</td>
      <td>0.9</td>
      <td>-86.7667</td>
      <td>34.745</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41</td>
      <td>0.05</td>
      <td>0.1</td>
      <td>-2.7738</td>
      <td>0.9</td>
      <td>-9.1822</td>
      <td>3.6346</td>
      <td>False</td>
    </tr>
    <tr>
      <td>42</td>
      <td>0.05</td>
      <td>0.15</td>
      <td>0.3714</td>
      <td>0.9</td>
      <td>-6.2036</td>
      <td>6.9463</td>
      <td>False</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.05</td>
      <td>0.2</td>
      <td>-0.986</td>
      <td>0.9</td>
      <td>-7.5166</td>
      <td>5.5447</td>
      <td>False</td>
    </tr>
    <tr>
      <td>44</td>
      <td>0.05</td>
      <td>0.25</td>
      <td>0.2294</td>
      <td>0.9</td>
      <td>-6.3801</td>
      <td>6.839</td>
      <td>False</td>
    </tr>
    <tr>
      <td>45</td>
      <td>0.06</td>
      <td>0.1</td>
      <td>23.237</td>
      <td>0.9</td>
      <td>-37.5302</td>
      <td>84.0042</td>
      <td>False</td>
    </tr>
    <tr>
      <td>46</td>
      <td>0.06</td>
      <td>0.15</td>
      <td>26.3822</td>
      <td>0.9</td>
      <td>-34.4028</td>
      <td>87.1671</td>
      <td>False</td>
    </tr>
    <tr>
      <td>47</td>
      <td>0.06</td>
      <td>0.2</td>
      <td>25.0248</td>
      <td>0.9</td>
      <td>-35.7554</td>
      <td>85.805</td>
      <td>False</td>
    </tr>
    <tr>
      <td>48</td>
      <td>0.06</td>
      <td>0.25</td>
      <td>26.2403</td>
      <td>0.9</td>
      <td>-34.5485</td>
      <td>87.029</td>
      <td>False</td>
    </tr>
    <tr>
      <td>49</td>
      <td>0.1</td>
      <td>0.15</td>
      <td>3.1452</td>
      <td>0.9</td>
      <td>-3.5337</td>
      <td>9.824</td>
      <td>False</td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>1.7879</td>
      <td>0.9</td>
      <td>-4.8474</td>
      <td>8.4231</td>
      <td>False</td>
    </tr>
    <tr>
      <td>51</td>
      <td>0.1</td>
      <td>0.25</td>
      <td>3.0033</td>
      <td>0.9</td>
      <td>-3.7096</td>
      <td>9.7161</td>
      <td>False</td>
    </tr>
    <tr>
      <td>52</td>
      <td>0.15</td>
      <td>0.2</td>
      <td>-1.3573</td>
      <td>0.9</td>
      <td>-8.1536</td>
      <td>5.4389</td>
      <td>False</td>
    </tr>
    <tr>
      <td>53</td>
      <td>0.15</td>
      <td>0.25</td>
      <td>-0.1419</td>
      <td>0.9</td>
      <td>-7.014</td>
      <td>6.7302</td>
      <td>False</td>
    </tr>
    <tr>
      <td>54</td>
      <td>0.2</td>
      <td>0.25</td>
      <td>1.2154</td>
      <td>0.9</td>
      <td>-5.6143</td>
      <td>8.0451</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot a universal confidence interval of each group mean comparing significant differences in group means. 
# Significant differences at the alpha=0.05 level can be identified by intervals that do not overlap 

oq_data = df_orderDetail['Quantity'].values
discount_labels = df_orderDetail['Discount'].values

from statsmodels.stats.multicomp import MultiComparison
oqd = MultiComparison(oq_data, discount_labels)
results = oqd.tukeyhsd()
results.plot_simultaneous(comparison_name=0.05, xlabel='Order Quantity', ylabel='Discount Level');
```


![png](/output/output_38_0.png)


### Effect Size
Calculate effect size using Cohen's D as well as any post-hoc tests.


```python
#### Cohen's d
def Cohen_d(group1, group2):
    # Compute Cohen's d.
    # group1: Series or NumPy array
    # group2: Series or NumPy array
    # returns a floating point number 
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d
```


```python
Cohen_d(discount, fullprice)
```




    0.2862724481729282



### Post-hoc Tests

The mean quantity per order is similar for each of the discount levels that we identified as significant. The obvious conclusion to draw from this is that offering a discount higher than 5% does not increase the order quantities; higher discounts only produce higher loss in revenue.


```python
# Extract revenue lost per discounted order where discount had no effect on order quantity
cur.execute("""SELECT Discount, 
                SUM(UnitPrice * Quantity) as 'revLoss',
                COUNT(OrderId) as 'NumOrders'
                FROM orderDetail  
                GROUP BY Discount
                HAVING Discount != 0 AND Discount != 0.05
                ORDER BY revLoss DESC;""")
df = pd.DataFrame(cur.fetchall())
df. columns = [i[0] for i in cur.description]
print(len(df))
df.head()

```

    9





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Discount</th>
      <th>revLoss</th>
      <th>NumOrders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.25</td>
      <td>131918.09</td>
      <td>154</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.20</td>
      <td>111476.38</td>
      <td>161</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.15</td>
      <td>102948.44</td>
      <td>157</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.10</td>
      <td>101665.71</td>
      <td>173</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.03</td>
      <td>124.65</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Total Revenue Forfeited $", df.revLoss.sum())
print("Number of Orders Affected ", df.NumOrders.sum())
print("Avg Forfeited Per Order $", df.revLoss.sum()/df.NumOrders.sum())
```

    Total Revenue Forfeited $ 448373.27
    Number of Orders Affected  653
    Avg Forfeited Per Order $ 686.6359418070444


## Analyze Results

Where alpha = 0.05, the null hypothesis is rejected. Discount amount has a statistically significant effect on the quantity in an order where the discount level is equal to 5%, 15%, 20% or 25%.

# H2: Country--Discount

**Do individual countries show a statistically significant preference for discount?**

**If so, which countries and to what extent?** 

## Hypotheses

- $H_0$: Countries purchase equal quantities of discounted vs non-discounted products.
- $H_A$: Countries purchase different quantities of discounted vs non-discounted products.

## EDA
Select the proper dataset for analysis, perform EDA, and generate data groups for testing.

### Select


```python
df_order = get_table(cur, "'Order'")
display(df_order.head())
display(df_orderDetail.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OrderId</th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>OrderDate</th>
      <th>RequiredDate</th>
      <th>ShippedDate</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10249</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>2012-08-16</td>
      <td>2012-07-10</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10250</td>
      <td>HANAR</td>
      <td>4</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-12</td>
      <td>2</td>
      <td>65.83</td>
      <td>Hanari Carnes</td>
      <td>Rua do Paço, 67</td>
      <td>Rio de Janeiro</td>
      <td>South America</td>
      <td>05454-876</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10251</td>
      <td>VICTE</td>
      <td>3</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-15</td>
      <td>1</td>
      <td>41.34</td>
      <td>Victuailles en stock</td>
      <td>2, rue du Commerce</td>
      <td>Lyon</td>
      <td>Western Europe</td>
      <td>69004</td>
      <td>France</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10252</td>
      <td>SUPRD</td>
      <td>4</td>
      <td>2012-07-09</td>
      <td>2012-08-06</td>
      <td>2012-07-11</td>
      <td>2</td>
      <td>51.30</td>
      <td>Suprêmes délices</td>
      <td>Boulevard Tirou, 255</td>
      <td>Charleroi</td>
      <td>Western Europe</td>
      <td>B-6000</td>
      <td>Belgium</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Rename 'Id' to 'OrderId' for joining tables with matching primary key name
df_order.rename({'Id':'OrderId'}, axis=1, inplace=True)
display(df_order.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OrderId</th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>OrderDate</th>
      <th>RequiredDate</th>
      <th>ShippedDate</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10249</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>2012-08-16</td>
      <td>2012-07-10</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10250</td>
      <td>HANAR</td>
      <td>4</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-12</td>
      <td>2</td>
      <td>65.83</td>
      <td>Hanari Carnes</td>
      <td>Rua do Paço, 67</td>
      <td>Rio de Janeiro</td>
      <td>South America</td>
      <td>05454-876</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10251</td>
      <td>VICTE</td>
      <td>3</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-15</td>
      <td>1</td>
      <td>41.34</td>
      <td>Victuailles en stock</td>
      <td>2, rue du Commerce</td>
      <td>Lyon</td>
      <td>Western Europe</td>
      <td>69004</td>
      <td>France</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10252</td>
      <td>SUPRD</td>
      <td>4</td>
      <td>2012-07-09</td>
      <td>2012-08-06</td>
      <td>2012-07-11</td>
      <td>2</td>
      <td>51.30</td>
      <td>Suprêmes délices</td>
      <td>Boulevard Tirou, 255</td>
      <td>Charleroi</td>
      <td>Western Europe</td>
      <td>B-6000</td>
      <td>Belgium</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_order.set_index('OrderId',inplace=True)
display(df_order.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>OrderDate</th>
      <th>RequiredDate</th>
      <th>ShippedDate</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
    </tr>
    <tr>
      <th>OrderId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10248</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <td>10249</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>2012-08-16</td>
      <td>2012-07-10</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
    </tr>
    <tr>
      <td>10250</td>
      <td>HANAR</td>
      <td>4</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-12</td>
      <td>2</td>
      <td>65.83</td>
      <td>Hanari Carnes</td>
      <td>Rua do Paço, 67</td>
      <td>Rio de Janeiro</td>
      <td>South America</td>
      <td>05454-876</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <td>10251</td>
      <td>VICTE</td>
      <td>3</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-15</td>
      <td>1</td>
      <td>41.34</td>
      <td>Victuailles en stock</td>
      <td>2, rue du Commerce</td>
      <td>Lyon</td>
      <td>Western Europe</td>
      <td>69004</td>
      <td>France</td>
    </tr>
    <tr>
      <td>10252</td>
      <td>SUPRD</td>
      <td>4</td>
      <td>2012-07-09</td>
      <td>2012-08-06</td>
      <td>2012-07-11</td>
      <td>2</td>
      <td>51.30</td>
      <td>Suprêmes délices</td>
      <td>Boulevard Tirou, 255</td>
      <td>Charleroi</td>
      <td>Western Europe</td>
      <td>B-6000</td>
      <td>Belgium</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_country = df_orderDetail.merge(df_order, on='OrderId', copy=True)
```

### Explore


```python
fs.ft.hakkeray.hot_stats(df_country, 'ShipCountry')
```

    -------->
    HOT!STATS
    <--------
    
    SHIPCOUNTRY
    Data Type: object
    
    min    Argentina
    max    Venezuela
    Name: ShipCountry, dtype: object 
    
    à-la-Mode: 
    0    USA
    dtype: object
    
    
    No Nulls Found!
    
    Non-Null Value Counts:
    USA            352
    Germany        328
    Brazil         203
    France         184
    UK             135
    Austria        125
    Venezuela      118
    Sweden          97
    Canada          75
    Mexico          72
    Belgium         56
    Ireland         55
    Spain           54
    Finland         54
    Italy           53
    Switzerland     52
    Denmark         46
    Argentina       34
    Portugal        30
    Poland          16
    Norway          16
    Name: ShipCountry, dtype: int64
    
    # Unique Values: 21
    


### Group


```python
countries = df_country.groupby('ShipCountry').groups
countries.keys()
```




    dict_keys(['Argentina', 'Austria', 'Belgium', 'Brazil', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Ireland', 'Italy', 'Mexico', 'Norway', 'Poland', 'Portugal', 'Spain', 'Sweden', 'Switzerland', 'UK', 'USA', 'Venezuela'])




```python
df_countries = df_country[['ShipCountry','Quantity','discounted']].copy()
df_countries.ShipCountry.value_counts()
```




    USA            352
    Germany        328
    Brazil         203
    France         184
    UK             135
    Austria        125
    Venezuela      118
    Sweden          97
    Canada          75
    Mexico          72
    Belgium         56
    Ireland         55
    Spain           54
    Finland         54
    Italy           53
    Switzerland     52
    Denmark         46
    Argentina       34
    Portugal        30
    Poland          16
    Norway          16
    Name: ShipCountry, dtype: int64




```python
import researchpy as rp
rp.summary_cont(df_countries.groupby(['discounted']))
```

    
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">Quantity</th>
    </tr>
    <tr>
      <th></th>
      <th>N</th>
      <th>Mean</th>
      <th>SD</th>
      <th>SE</th>
      <th>95% Conf.</th>
      <th>Interval</th>
    </tr>
    <tr>
      <th>discounted</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1317</td>
      <td>21.715262</td>
      <td>17.507493</td>
      <td>0.482426</td>
      <td>20.769706</td>
      <td>22.660818</td>
    </tr>
    <tr>
      <td>1</td>
      <td>838</td>
      <td>27.109785</td>
      <td>20.771439</td>
      <td>0.717537</td>
      <td>25.703412</td>
      <td>28.516159</td>
    </tr>
  </tbody>
</table>
</div>



## Test

### Sample Size


```python
# Check if sample sizes allow us to ignore assumptions;
# visualize sample size comparisons for two groups (normality check)

stat_dict = {}

for k,v in countries.items():
    try:
        grp0 = df_countries.loc[v].groupby('discounted').get_group(0)['Quantity']
        grp1 = df_countries.loc[v].groupby('discounted').get_group(1)['Quantity']
        print(f"{k}")
        
        import scipy.stats as stat

        plt.bar(x='Full Price', height=grp0.mean(), yerr=stat.sem(grp0))
        plt.bar(x='Discounted', height=grp1.mean(), yerr=stat.sem(grp1))
        plt.show()
        
    except:
        pass
        
    try:
        result = stat.ttest_ind(grp0,grp1)
        if result[1] < 0.05:
            stat_dict[k] = result[1]
            print(f"\n{k} PREFERS DISCOUNTS!")
        else:
            continue
    except:
        print(f"{k} does not contain one of the groups.")
stat_dict
```

    Argentina does not contain one of the groups.
    Austria



![png](/output/output_64_1.png)


    Belgium



![png](/output/output_64_3.png)


    Brazil



![png](/output/output_64_5.png)


    Canada



![png](/output/output_64_7.png)


    
    Canada PREFERS DISCOUNTS!
    Denmark



![png](/output/output_64_9.png)


    Finland



![png](/output/output_64_11.png)


    France



![png](/output/output_64_13.png)


    Germany



![png](/output/output_64_15.png)


    Ireland



![png](/output/output_64_17.png)


    Italy



![png](/output/output_64_19.png)


    Mexico



![png](/output/output_64_21.png)


    
    Norway PREFERS DISCOUNTS!
    Portugal



![png](/output/output_64_23.png)


    Spain



![png](/output/output_64_25.png)


    
    Spain PREFERS DISCOUNTS!
    Sweden



![png](/output/output_64_27.png)


    Switzerland



![png](/output/output_64_29.png)


    UK



![png](/output/output_64_31.png)


    
    UK PREFERS DISCOUNTS!
    USA



![png](/output/output_64_33.png)


    
    USA PREFERS DISCOUNTS!
    Venezuela



![png](/output/output_64_35.png)





    {'Canada': 0.0010297982736886485,
     'Norway': 0.04480094051665529,
     'Spain': 0.0025087181106716217,
     'UK': 0.00031794803200322925,
     'USA': 0.019868707223971476}




```python
stat_dict
```




    {'Canada': 0.0010297982736886485,
     'Norway': 0.04480094051665529,
     'Spain': 0.0025087181106716217,
     'UK': 0.00031794803200322925,
     'USA': 0.019868707223971476}



### Normality Test


```python
fig = plt.figure(figsize=(10,8))
ax = fig.gca(title="Distribution of Full price vs Discounted Orders")

sns.distplot(grp0)
sns.distplot(grp1)
ax.legend(['Full Price','Discounted'])
```




    <matplotlib.legend.Legend at 0x1a25ab2978>




![png](/output/output_67_1.png)



```python
# Test for normality - D'Agostino-Pearson's normality test: scipy.stats.normaltest
stat.normaltest(grp0), stat.normaltest(grp1)
```




    (NormaltestResult(statistic=9.316225653095811, pvalue=0.009484344125890621),
     NormaltestResult(statistic=10.255309993341813, pvalue=0.005930451108115991))




```python
# Run non-parametric test (since normality test failed)
stat.mannwhitneyu(grp0, grp1)
```




    MannwhitneyuResult(statistic=1632.5, pvalue=0.44935140740973323)



**Canada, Spain, UK and the USA have pvalues < 0.05 indicating there is a relationship between discount and order quantity and the null hypothesis is rejected for these individual countries.**

### Statistical Test


```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Quantity~C(discounted)+C(ShipCountry)+C(discounted):C(ShipCountry)", data=df_countries).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

    /Users/hakkeray/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/base/model.py:1752: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 20, but rank is 18
      'rank is %d' % (J, J_), ValueWarning)
    /Users/hakkeray/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/base/model.py:1752: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 20, but rank is 18
      'rank is %d' % (J, J_), ValueWarning)



```python
# reformat scientific notation of results for easier interpretation
anova_table.style.format("{:.5f}", subset=['PR(>F)'])
```




<style  type="text/css" >
</style><table id="T_3d92bd06_29bb_11ea_af86_f40f2405a054" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sum_sq</th>        <th class="col_heading level0 col1" >df</th>        <th class="col_heading level0 col2" >F</th>        <th class="col_heading level0 col3" >PR(>F)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_3d92bd06_29bb_11ea_af86_f40f2405a054level0_row0" class="row_heading level0 row0" >C(discounted)</th>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row0_col0" class="data row0 col0" >9.78092e-08</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row0_col1" class="data row0 col1" >1</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row0_col2" class="data row0 col2" >3.07557e-10</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row0_col3" class="data row0 col3" >0.99999</td>
            </tr>
            <tr>
                        <th id="T_3d92bd06_29bb_11ea_af86_f40f2405a054level0_row1" class="row_heading level0 row1" >C(ShipCountry)</th>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row1_col0" class="data row1 col0" >101347</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row1_col1" class="data row1 col1" >20</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row1_col2" class="data row1 col2" >15.9341</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row1_col3" class="data row1 col3" >0.00000</td>
            </tr>
            <tr>
                        <th id="T_3d92bd06_29bb_11ea_af86_f40f2405a054level0_row2" class="row_heading level0 row2" >C(discounted):C(ShipCountry)</th>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row2_col0" class="data row2 col0" >15584.9</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row2_col1" class="data row2 col1" >20</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row2_col2" class="data row2 col2" >2.4503</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row2_col3" class="data row2 col3" >0.00061</td>
            </tr>
            <tr>
                        <th id="T_3d92bd06_29bb_11ea_af86_f40f2405a054level0_row3" class="row_heading level0 row3" >Residual</th>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row3_col0" class="data row3 col0" >672930</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row3_col1" class="data row3 col1" >2116</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row3_col2" class="data row3 col2" >nan</td>
                        <td id="T_3d92bd06_29bb_11ea_af86_f40f2405a054row3_col3" class="data row3 col3" >nan</td>
            </tr>
    </tbody></table>




```python
# calculate ttest_ind p-values and significance for individual countries
print(f"\n Countries with p-values < 0.05 - Null Hypothesis Rejected:")
for k,v in countries.items():
    try:
        grp0 = df_countries.loc[v].groupby('discounted').get_group(0)['Quantity']
        grp1 = df_countries.loc[v].groupby('discounted').get_group(1)['Quantity']
        result = stat.ttest_ind(grp0,grp1)
        if result[1] < 0.05:
            
            print(f"\n\t{k}: {result[1].round(4)}")
        else:
            continue
    except:
        None 
```

    
     Countries with p-values < 0.05 - Null Hypothesis Rejected:
    
    	Canada: 0.001
    
    	Spain: 0.0025
    
    	UK: 0.0003
    
    	USA: 0.0199


Although discount does not have a significant effect on countries overall (p = 0.99), there is a statistically significant relationship between order quantities and discount in some of the countries (p=0.0006).

Countries with p-values < 0.05 - Null Hypothesis Rejected:

	Canada: 0.001

	Spain: 0.0025

	UK: 0.0003

	USA: 0.0199


```python
y1 = df_countries.groupby('discounted').get_group(1)['Quantity']


fig = plt.figure(figsize=(18,12))
ax = fig.gca()

ax = sns.barplot(x='ShipCountry', y=y1, data=df_countries)

ax.set_title('Average Discount Order Quantity by Country', fontdict={'family': 'PT Mono', 'size':16})
```




    Text(0.5, 1.0, 'Average Discount Order Quantity by Country')




![png](/output/output_76_1.png)


### Effect Size

Effect size testing is unnecessary since the null hypothesis for the main question was not rejected.

### Post-hoc Tests


```python
#!pip install pandasql
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
```


```python
# Compare number of discount vs fullprice orders by country.
# Create bar plots grouped as discount vs fullprice orders by country
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18,8))

q1 = "SELECT ShipCountry, AVG(Quantity) as OrderQty from df_countries where discounted = 0 group by 1;"
q2 = "SELECT ShipCountry, AVG(Quantity) as OrderQty from df_countries where discounted = 1 group by 1;"

df_fpCount = pysqldf(q1)
df_dcCount = pysqldf(q2)

df_fpCount['Group'] = 'FullPrice'
df_dcCount['Group'] = 'Discount'

df_country_qty = pd.concat([df_fpCount, df_dcCount], axis=0)

display(df_country_qty.describe())

#ax = sns.barplot(x='ShipCountry', y='NumOrders', data=country_df, hue='Group', palette='pastel', orient='v')
#ax.set_title('Number of Fullprice vs Discount Orders by Country', fontdict={'family': 'monospace', 'size':16})

#ax1 = sns.barplot(x='ShipCountry', y='TotalQty', data=country_df, hue='Group', palette='pastel', orient='v')
#ax1.set_title('Total Qty of Fullprice vs Discount Orders by Country', fontdict={'family': 'monospace', 'size':16})

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18,8))
ax = fig.gca(title="Average Order Quantity by Country: Fullprice vs Discount")

sns.barplot(x='ShipCountry', y='OrderQty', ax=ax, data=df_country_qty, hue='Group', 
            palette='pastel', orient='v', ci=68, capsize=.2)

## Set Title,X/Y Labels,fonts,formatting
ax_font = {'family':'monospace','weight':'semibold','size':14}
tick_font = {'size':12,'ha':'center','rotation':45}
t_label = "Average Order Quantity by Country: Fullprice vs Discount Orders"
t_font = {'family': 'PT Mono', 'size':18}

ax.set_ylabel("Order Qty", fontdict=ax_font)
ax.set_xlabel("Country", fontdict=ax_font)
#ax.set_title('Average Order Quantity by Country: Fullprice vs Discount', fontdict={'family': 'PT Mono', 'size':16})
ax.set_title(t_label, fontdict=t_font)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OrderQty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>22.596281</td>
    </tr>
    <tr>
      <td>std</td>
      <td>7.620086</td>
    </tr>
    <tr>
      <td>min</td>
      <td>9.970588</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>17.458458</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>21.750000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>25.985539</td>
    </tr>
    <tr>
      <td>max</td>
      <td>43.172414</td>
    </tr>
  </tbody>
</table>
</div>





    Text(0.5, 1.0, 'Average Order Quantity by Country: Fullprice vs Discount Orders')




![png](/output/output_81_2.png)


According to the plot above, the actual number of discounted orders is lower than the number of full price orders. Let's compare the sum of quantities for these orders in each group.


```python
# Compare number of discount vs fullprice orders by country.
# Create bar plots grouped as discount vs fullprice orders by country
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18,8))

q1 = "SELECT ShipCountry, Count(*) as OrderCount from df_countries where discounted = 0 group by 1;"
q2 = "SELECT ShipCountry, Count(*) as OrderCount from df_countries where discounted = 1 group by 1;"

df_fpCount = pysqldf(q1)
df_dcCount = pysqldf(q2)

df_fpCount['Group'] = 'FullPrice'
df_dcCount['Group'] = 'Discount'

df_country_count = pd.concat([df_fpCount, df_dcCount], axis=0)

display(df_country_count.describe())

#ax = sns.barplot(x='ShipCountry', y='NumOrders', data=country_df, hue='Group', palette='pastel', orient='v')
#ax.set_title('Number of Fullprice vs Discount Orders by Country', fontdict={'family': 'monospace', 'size':16})

#ax1 = sns.barplot(x='ShipCountry', y='TotalQty', data=country_df, hue='Group', palette='pastel', orient='v')
#ax1.set_title('Total Qty of Fullprice vs Discount Orders by Country', fontdict={'family': 'monospace', 'size':16})


fig = plt.figure(figsize=(18,8))
ax = fig.gca(title="Mean QPO by Country")

sns.barplot(x='ShipCountry', y='OrderCount', ax=ax, data=df_country_count, hue='Group', palette='Reds_d', 
            orient='v', ci=68, capsize=.2)

## Set Title,X/Y Labels,fonts,formatting
ax_font = {'family':'monospace','weight':'semibold','size':14}
tick_font = {'size':12,'ha':'center','rotation':45}
t_label = "Count of Fullprice vs Discount Orders by Country"
t_font = {'family': 'PT Mono', 'size':18}

ax.set_ylabel("Number of Orders", fontdict=ax_font)
ax.set_xlabel("Country", fontdict=ax_font)
#ax.set_title('Average Order Quantity by Country: Fullprice vs Discount', fontdict={'family': 'PT Mono', 'size':16})
ax.set_title(t_label, fontdict=t_font)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OrderCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>55.256410</td>
    </tr>
    <tr>
      <td>std</td>
      <td>48.722478</td>
    </tr>
    <tr>
      <td>min</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>69.500000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>210.000000</td>
    </tr>
  </tbody>
</table>
</div>





    Text(0.5, 1.0, 'Count of Fullprice vs Discount Orders by Country')




![png](/output/output_83_2.png)


This still doesn't tell us much about whether or not these countries prefer discounts (tend to order more products) or not - in order to get better insight, we need to look at the average order size (mean quantities per order) for each group.


```python
# Compare number of discount vs fullprice orders by country.
# Create bar plots grouped as discount vs fullprice orders by country
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18,8))

#q1 = "SELECT ShipCountry, Count(*) as NumOrders, SUM(Quantity) as TotalQty, AVG(Quantity) as MeanQPO from df_countries where discounted = 0 group by 1;"
#q2 = "SELECT ShipCountry, Count(*) as NumOrders, SUM(Quantity) as TotalQty, AVG(Quantity) as MeanQPO from df_countries where discounted = 1 group by 1;"

q1 = "SELECT ShipCountry, AVG(Quantity) as MeanQPO from df_countries where discounted = 0 group by 1;"
q2 = "SELECT ShipCountry, AVG(Quantity) as MeanQPO from df_countries where discounted = 1 group by 1;"

fullprice_df = pysqldf(q1)
discount_df = pysqldf(q2)

fullprice_df['Group'] = 'FullPrice'
discount_df['Group'] = 'Discount'

country_df = pd.concat([fullprice_df, discount_df], axis=0)

display(country_df.describe())

#ax = sns.barplot(x='ShipCountry', y='NumOrders', data=country_df, hue='Group', palette='pastel', orient='v')
#ax.set_title('Number of Fullprice vs Discount Orders by Country', fontdict={'family': 'monospace', 'size':16})

#ax1 = sns.barplot(x='ShipCountry', y='TotalQty', data=country_df, hue='Group', palette='pastel', orient='v')
#ax1.set_title('Total Qty of Fullprice vs Discount Orders by Country', fontdict={'family': 'monospace', 'size':16})


fig = plt.figure(figsize=(18,8))
ax = fig.gca(title="Mean QPO by Country")

sns.barplot(x='ShipCountry', y='MeanQPO', ax=ax, data=country_df, hue='Group', palette='Greens_d', 
            orient='v', capsize=.2)

## Set Title,X/Y Labels,fonts,formatting
ax_font = {'family':'monospace','weight':'semibold','size':14}
tick_font = {'size':12,'ha':'center','rotation':45}
t_label = "Average Order Quantity by Country: Fullprice vs Discount"
t_font = {'family': 'PT Mono', 'size':18}

ax.set_ylabel("Avg Qty per Order ", fontdict=ax_font)
ax.set_xlabel("Country", fontdict=ax_font)
#ax.set_title('Average Order Quantity by Country: Fullprice vs Discount', fontdict={'family': 'PT Mono', 'size':16})
ax.set_title(t_label, fontdict=t_font)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MeanQPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>22.596281</td>
    </tr>
    <tr>
      <td>std</td>
      <td>7.620086</td>
    </tr>
    <tr>
      <td>min</td>
      <td>9.970588</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>17.458458</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>21.750000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>25.985539</td>
    </tr>
    <tr>
      <td>max</td>
      <td>43.172414</td>
    </tr>
  </tbody>
</table>
</div>





    Text(0.5, 1.0, 'Average Order Quantity by Country: Fullprice vs Discount')




![png](/output/output_85_2.png)


The above plots indicate that when a discount is offered, certain countries order higher quantities of products. Let's look at the values to determine what percentage more they purchase when an order is discounted.


```python
# add new col for countries where discount has significant effect
fig = plt.figure(figsize=(18,12))
ax = fig.gca()
df_countries['effect_cqd'] = df_countries['ShipCountry'].isin(['Spain', 'UK', 'USA', 'Canada'])
ax = sns.barplot(x='ShipCountry', y='Quantity', hue='effect_cqd', palette='pastel', data=df_countries)

```


![png](/output/output_87_0.png)



```python
q1 = "SELECT ShipCountry, Count(*) as OrderCount from df_countries where discounted = 0 group by 1;"
q2 = "SELECT ShipCountry, Count(*) as OrderCount from df_countries where discounted = 1 group by 1;"

df_fpCount = pysqldf(q1)
df_dcCount = pysqldf(q2)

df_fpCount['Group'] = 'FullPrice'
df_dcCount['Group'] = 'Discount'

df_countryCount = pd.concat([df_fpCount, df_dcCount])

fig = plt.figure(figsize=(18,8))
ax = fig.gca(title="Average Order Quantity by Country")

ax = sns.barplot(x='ShipCountry', y='OrderCount', data=df_countryCount)
ax.set_title('Order Count by Country', fontdict={'family': 'PT Mono', 'size':16})
```




    Text(0.5, 1.0, 'Order Count by Country')




![png](/output/output_88_1.png)


## Results

For certain individual countries (Spain, Canada, UK, USA), the null hypothesis is rejected with 95% certainty (alpha=0.05) 

# H3: Region & Revenue

**Does average revenue per order vary between different customer regions?**

**If so, how do the regions rank in terms of average revenue per order?**

*Additional questions to explore:*
**Does geographic distance between distributor and shipcountry have an effect on order quantity?**
**Does shipping cost have an effect on order quantity?**

## Hypotheses

$H_0$ the average revenue per order is the same between different customer regions.

$H_1$ Alternate hypothesis: the average revenue per order is different (higher or lower) across different customer regions.

*The alpha level (i.e. the probability of rejecting the null hypothesis when it is true) is = 0.05.*

## EDA

Select the proper dataset for analysis, generate data groups for testing, perform EDA.

### Select


```python
# Extract revenue per product per order
cur.execute("""SELECT c.Region, od.OrderId, od.Quantity, od.UnitPrice, od.Discount
FROM Customer c
JOIN 'Order' o ON c.Id = o.CustomerId
JOIN OrderDetail od USING(OrderId);""")
df = pd.DataFrame(cur.fetchall())
df. columns = [i[0] for i in cur.description]
print(len(df))
df.head()
```

    2078





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>OrderId</th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Western Europe</td>
      <td>10248</td>
      <td>12</td>
      <td>14.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Western Europe</td>
      <td>10248</td>
      <td>10</td>
      <td>9.8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Western Europe</td>
      <td>10248</td>
      <td>5</td>
      <td>34.8</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Western Europe</td>
      <td>10249</td>
      <td>9</td>
      <td>18.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Western Europe</td>
      <td>10249</td>
      <td>40</td>
      <td>42.4</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get total revenue per order

df['Revenue'] = df.Quantity * df.UnitPrice * (1-df.Discount)
```


```python
# Drop unnecessary columns
df.drop(['Quantity', 'UnitPrice', 'Discount'], axis=1, inplace=True)
```

### Group


```python
# Group data by order and get average revenue per order for each region
df_region = df.groupby(['Region', 'OrderId'])['Revenue'].mean().reset_index()
# drop Order Id (no longer necessary)
df_region.drop('OrderId', axis=1, inplace=True)
# check changes
df_region.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>British Isles</td>
      <td>239.70</td>
    </tr>
    <tr>
      <td>1</td>
      <td>British Isles</td>
      <td>661.25</td>
    </tr>
    <tr>
      <td>2</td>
      <td>British Isles</td>
      <td>352.40</td>
    </tr>
    <tr>
      <td>3</td>
      <td>British Isles</td>
      <td>258.40</td>
    </tr>
    <tr>
      <td>4</td>
      <td>British Isles</td>
      <td>120.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Explore sample sizes before testing: n > 30 to pass assumptions
df_region.groupby('Region').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Revenue</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>British Isles</td>
      <td>75</td>
    </tr>
    <tr>
      <td>Central America</td>
      <td>21</td>
    </tr>
    <tr>
      <td>Eastern Europe</td>
      <td>7</td>
    </tr>
    <tr>
      <td>North America</td>
      <td>152</td>
    </tr>
    <tr>
      <td>Northern Europe</td>
      <td>55</td>
    </tr>
    <tr>
      <td>Scandinavia</td>
      <td>28</td>
    </tr>
    <tr>
      <td>South America</td>
      <td>127</td>
    </tr>
    <tr>
      <td>Southern Europe</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Western Europe</td>
      <td>272</td>
    </tr>
  </tbody>
</table>
</div>



Some of the sample sizes are too small to ignore assumptions of normality. We can combine some regions to meet the required threshold of n > 30.


```python
# Group sub-regions together to create sample sizes adequately large for ANOVA testing  (min 30)

# Group Scandinavia, Northern and Eastern Europe
df_region.loc[(df_region.Region == 'Scandinavia') | (df_region.Region == 'Eastern Europe') | (df_region.Region == 'Northern Europe'), 'Region'] = 'North Europe'

# Group South and Central America
df_region.loc[(df_region.Region == 'South America') | (df_region.Region == 'Central America'), 'Region'] = 'South Americas'

# Review sizes of new groups
df_region.groupby('Region').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Revenue</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>British Isles</td>
      <td>75</td>
    </tr>
    <tr>
      <td>North America</td>
      <td>152</td>
    </tr>
    <tr>
      <td>North Europe</td>
      <td>90</td>
    </tr>
    <tr>
      <td>South Americas</td>
      <td>148</td>
    </tr>
    <tr>
      <td>Southern Europe</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Western Europe</td>
      <td>272</td>
    </tr>
  </tbody>
</table>
</div>



### Explore


```python
fig = plt.figure(figsize=(10,8))
ax = fig.gca()

sns.distplot(grp0)
sns.distplot(grp1)
ax.legend(['Full Price','Discounted'])

# Plot number of orders, total revenue, and average revenue per order by region
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,8))
# Number of orders
df_region.groupby(['Region'])['Revenue'].count().plot(kind='barh', ax=ax1, color='b')

# Total Revenue
df_region.groupby(['Region'])['Revenue'].sum().plot(kind='barh', ax=ax2, color='r')

# Average Revenue
df_region.groupby(['Region'])['Revenue'].mean().plot(kind='barh', ax=ax3, color='g')

# Label plots and axes
ax1.set_title('Total Orders')
ax1.set_ylabel('')
ax2.set_title('Total Revenue in US$')
ax2.set_ylabel('')
ax3.set_title('Average Revenue per Order US$')
ax3.set_ylabel('')

fig.subplots_adjust(hspace=0.4);
```


![png](/output/output_104_0.png)



![png](/output/output_104_1.png)


The graphs show that Western Europe is the region with the greatest number of orders, and also has the greatest total revenue. However, North America has the most expensive order on average (followed by Western Europe). Southern and Eastern Europe has the lowest number of orders, lowest total revenue, and cheapest order on average. The third graph lent support to the alternate hypothesis that there are significant differences in average order revenue between regions. 

## Test

### Sample Size
Check if sample sizes allow us to ignore assumptions of normality


```python
# visualize sample size comparisons, check normality (pvals)
fig = plt.figure(figsize=(12,6))
ax = fig.gca()

ax = sns.barplot(x='Region', y='Revenue', data=df_region, ci=68, palette="pastel", hue='Region')
ax.set_title('Average Order Revenue by Region', fontdict={'family': 'PT Mono', 'size':16})
```




    Text(0.5, 1.0, 'Average Order Revenue by Region')




![png](/output/output_108_1.png)


### Normality


```python

```

### Statistical


```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Revenue~C(Region)+Revenue:C(Region)", data=df_region).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
# reformat scientific notation of results for easier interpretation
anova_table.style.format("{:.5f}", subset=['PR(>F)'])
```




<style  type="text/css" >
</style><table id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sum_sq</th>        <th class="col_heading level0 col1" >df</th>        <th class="col_heading level0 col2" >F</th>        <th class="col_heading level0 col3" >PR(>F)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054level0_row0" class="row_heading level0 row0" >C(Region)</th>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row0_col0" class="data row0 col0" >1.03486e+07</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row0_col1" class="data row0 col1" >5</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row0_col2" class="data row0 col2" >3.98262e+30</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row0_col3" class="data row0 col3" >0.00000</td>
            </tr>
            <tr>
                        <th id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054level0_row1" class="row_heading level0 row1" >Revenue:C(Region)</th>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row1_col0" class="data row1 col0" >5.34162e+08</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row1_col1" class="data row1 col1" >6</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row1_col2" class="data row1 col2" >1.71309e+32</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row1_col3" class="data row1 col3" >0.00000</td>
            </tr>
            <tr>
                        <th id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054level0_row2" class="row_heading level0 row2" >Residual</th>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row2_col0" class="data row2 col0" >4.10034e-22</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row2_col1" class="data row2 col1" >789</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row2_col2" class="data row2 col2" >nan</td>
                        <td id="T_60364b3e_29bb_11ea_9ec9_f40f2405a054row2_col3" class="data row2 col3" >nan</td>
            </tr>
    </tbody></table>




```python
# run tukey test for OQD (Order Quantity Discount) 
data = df_region['Revenue'].values
labels = df_region['Region'].values

import statsmodels.api as sms
model = sms.stats.multicomp.pairwise_tukeyhsd(data,labels)

# save OQD tukey test model results into dataframe (OQD: order quantity discount)
tukey_OQD = pd.DataFrame(data=model._results_table[1:], columns=model._results_table[0])
tukey_OQD
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>British Isles</td>
      <td>North America</td>
      <td>116.4615</td>
      <td>0.9</td>
      <td>-213.9625</td>
      <td>446.8854</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>British Isles</td>
      <td>North Europe</td>
      <td>-88.1693</td>
      <td>0.9</td>
      <td>-454.2704</td>
      <td>277.9318</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>British Isles</td>
      <td>South Americas</td>
      <td>-84.2501</td>
      <td>0.9</td>
      <td>-416.146</td>
      <td>247.6458</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>British Isles</td>
      <td>Southern Europe</td>
      <td>-271.7815</td>
      <td>0.3745</td>
      <td>-670.2535</td>
      <td>126.6904</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>British Isles</td>
      <td>Western Europe</td>
      <td>81.6889</td>
      <td>0.9</td>
      <td>-223.7052</td>
      <td>387.083</td>
      <td>False</td>
    </tr>
    <tr>
      <td>5</td>
      <td>North America</td>
      <td>North Europe</td>
      <td>-204.6308</td>
      <td>0.4191</td>
      <td>-516.0716</td>
      <td>106.8101</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>North America</td>
      <td>South Americas</td>
      <td>-200.7115</td>
      <td>0.2778</td>
      <td>-471.1191</td>
      <td>69.6961</td>
      <td>False</td>
    </tr>
    <tr>
      <td>7</td>
      <td>North America</td>
      <td>Southern Europe</td>
      <td>-388.243</td>
      <td>0.0191</td>
      <td>-737.1631</td>
      <td>-39.3228</td>
      <td>True</td>
    </tr>
    <tr>
      <td>8</td>
      <td>North America</td>
      <td>Western Europe</td>
      <td>-34.7725</td>
      <td>0.9</td>
      <td>-271.9032</td>
      <td>202.3581</td>
      <td>False</td>
    </tr>
    <tr>
      <td>9</td>
      <td>North Europe</td>
      <td>South Americas</td>
      <td>3.9193</td>
      <td>0.9</td>
      <td>-309.0829</td>
      <td>316.9214</td>
      <td>False</td>
    </tr>
    <tr>
      <td>10</td>
      <td>North Europe</td>
      <td>Southern Europe</td>
      <td>-183.6122</td>
      <td>0.7177</td>
      <td>-566.4899</td>
      <td>199.2655</td>
      <td>False</td>
    </tr>
    <tr>
      <td>11</td>
      <td>North Europe</td>
      <td>Western Europe</td>
      <td>169.8582</td>
      <td>0.5251</td>
      <td>-114.889</td>
      <td>454.6055</td>
      <td>False</td>
    </tr>
    <tr>
      <td>12</td>
      <td>South Americas</td>
      <td>Southern Europe</td>
      <td>-187.5315</td>
      <td>0.6259</td>
      <td>-537.8459</td>
      <td>162.783</td>
      <td>False</td>
    </tr>
    <tr>
      <td>13</td>
      <td>South Americas</td>
      <td>Western Europe</td>
      <td>165.939</td>
      <td>0.3541</td>
      <td>-73.2385</td>
      <td>405.1164</td>
      <td>False</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Southern Europe</td>
      <td>Western Europe</td>
      <td>353.4704</td>
      <td>0.0242</td>
      <td>28.1539</td>
      <td>678.787</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



North America and Southern Europe:  pval = 0.01, mean diff: -388.24

Southern Europe and Western Europe: pval = 0.02, mean diff: 353.4704

### Effect Size

Cohen's D


```python
northamerica = df_region.loc[df_region['Region'] == 'North America']
southerneurope = df_region.loc[df_region['Region'] == 'Southern Europe']
westerneurope = df_region.loc[df_region['Region'] == 'Western Europe']

na_se = Cohen_d(northamerica.Revenue, southerneurope.Revenue)
se_we = Cohen_d(southerneurope.Revenue, westerneurope.Revenue)

print(na_se, se_we)
```

    0.5891669383438923 -0.5462384714677272


## Post-Hoc Tests


```python
# log-transforming revenue per order
logRegion_df = df_region.copy()
logRegion_df['Revenue'] = np.log(df_region['Revenue'])

# Plotting the distributions for the log-transformed data
sns.set_style("whitegrid")

fig = plt.figure(figsize=(12,8))
ax = fig.gca(title="Distribution of Revenue Per Order by Region")

for region in set(logRegion_df.Region):
    region_group = logRegion_df.loc[logRegion_df['Region'] == region]
    sns.distplot(region_group['Revenue'], hist_kws=dict(alpha=0.5), label=region)
    ax.legend()
    ax.set_label('Revenue per Order (log-transformed)')
```


![png](/output/output_118_0.png)



```python
# The data is more normally distributed, and variances from the mean were more similar. 
# run an ANOVA test:

# Fitting a model of revenue per order on Region categories - ANOVA table
lm = ols('Revenue ~ C(Region)', logRegion_df).fit()
sm.stats.anova_lm(lm, typ=2)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C(Region)</td>
      <td>48.004167</td>
      <td>5.0</td>
      <td>12.076998</td>
      <td>2.713885e-11</td>
    </tr>
    <tr>
      <td>Residual</td>
      <td>631.999979</td>
      <td>795.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Results

At an alpha level of 0.05 significance, revenue does vary between regions and therefore the null hypothesis is rejected.



The ANOVA table above revealed that the p-value is lower than the alpha value of 0.05. Therefore I was able to reject the null hypothesis and accept the alternate hypothesis. There are statistically significant differences in average order value between different regions, i.e. customers from different parts of the world spend different amounts of money on their orders, on average.
Conclusions
Business insights:
There are statistically significant differences in the average revenue per order from customers from different regions.
Western European customers place the most orders, and are the single biggest contributors to Northwind’s bottom line. However, although North American customers have placed roughly half as many orders as those from Western Europe, they spend more per order, on average.
The difference between the region with the most expensive orders on average (North America, $1,945.93) and the region with the least expensive orders (Southern and Eastern Europe, $686.73) is $1,259.20, or 2.8 times more for orders from North America.
Southern and Eastern Europe has the smallest number of orders, the lowest total revenue, and the lowest average revenue per order.
North American customers have placed a similar number of orders to those from South and Central America, but their average expenditure per order is 1.8 times higher.
Potential business actions and directions for future work:
If Northwind was looking to focus on more profitable customers, a potential action would be to stop serving customers in Southern and Eastern Europe, and to focus more on customers in Western Europe and North America.
However, further analysis would be needed to confirm these findings. For example, it might be the case that some more expensive products are only available in certain regions.

---

# H4: Season+Quantity:ProductCategory

1: **Does time of year (month) have an effect on order quantity overall?**

2: **Does time of year (month) have an effect on order quantity of specific product categories?**

3: **Does time of year (month) have an effect on order quantity by region?**

## Hypotheses
    
* $𝐻_1$  : Time of year has a statistically significant effect on average quantity per order.

* $𝐻_0$ : Time of year has no relationship with average quantity per order.

## EDA
- Select proper dataset for analysis: orderDetail, order
- Generate data groups for testing: number of orders per month, order quantity per month
- Explore data (sample sizes, distribution/density)

### Select


```python
df_months = df_orderDetail.merge(df_order, on='OrderId', copy=True)
df_months.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>OrderDate</th>
      <th>RequiredDate</th>
      <th>ShippedDate</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>0</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>2012-08-16</td>
      <td>2012-07-10</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>0</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>2012-08-16</td>
      <td>2012-07-10</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.to_datetime(df_months['OrderDate'], format='%Y/%m/%d').head()
```




    0   2012-07-04
    1   2012-07-04
    2   2012-07-04
    3   2012-07-05
    4   2012-07-05
    Name: OrderDate, dtype: datetime64[ns]




```python
df_months['OrderMonth'] = pd.DatetimeIndex(df_months['OrderDate']).month
df_months['OrderYear'] = pd.DatetimeIndex(df_months['OrderDate']).year
df_months.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>OrderDate</th>
      <th>...</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
      <th>OrderMonth</th>
      <th>OrderYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>...</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>...</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>...</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>0</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>...</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>0</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>...</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
      <td>7</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df_months.set_index('OrderDate', inplace=True)
df_months.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>discounted</th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>RequiredDate</th>
      <th>...</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
      <th>OrderMonth</th>
      <th>OrderYear</th>
    </tr>
    <tr>
      <th>OrderDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2012-07-04</td>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-08-01</td>
      <td>...</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-04</td>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-08-01</td>
      <td>...</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-04</td>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-08-01</td>
      <td>...</td>
      <td>3</td>
      <td>32.38</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-05</td>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>0</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-08-16</td>
      <td>...</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-05</td>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>0</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-08-16</td>
      <td>...</td>
      <td>1</td>
      <td>11.61</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
      <td>7</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Group


```python
# create seasonal-based dataframe with only columns we need
#keep_cols = ['OrderId', 'ProductId', 'UnitPrice', 'Quantity', 'ShipCountry', 'OrderMonth', 'OrderYear', 'Season']
drop_cols = ['OrderId', 'discounted', 'CustomerId', 'EmployeeId', 'Freight', 'RequiredDate', 'ShippedDate', 'ShipVia', 'ShipName', 'ShipAddress', 'ShipCity', 'ShipPostalCode']
df_monthly = df_months.copy()
df_monthly.drop(drop_cols, axis=1, inplace=True)
df_monthly.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>ShipRegion</th>
      <th>ShipCountry</th>
      <th>OrderMonth</th>
      <th>OrderYear</th>
    </tr>
    <tr>
      <th>OrderDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2012-07-04</td>
      <td>10248/11</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>Western Europe</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-04</td>
      <td>10248/42</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>Western Europe</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-04</td>
      <td>10248/72</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>Western Europe</td>
      <td>France</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-05</td>
      <td>10249/14</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>Western Europe</td>
      <td>Germany</td>
      <td>7</td>
      <td>2012</td>
    </tr>
    <tr>
      <td>2012-07-05</td>
      <td>10249/51</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>Western Europe</td>
      <td>Germany</td>
      <td>7</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
</div>




```python
meanqpo = df_monthly.groupby('OrderMonth')['Quantity'].mean()
```

### Explore

## Test

### Sample Size


```python
sns.set_style("whitegrid")
%config InlineBackend.figure_format='retina'
%matplotlib inline


# Check if sample sizes allow us to ignore assumptions;
# visualize sample size comparisons for two groups (normality check)
fig = plt.figure(figsize=(14,8))
ax = fig.gca()
ax = sns.barplot(x='OrderMonth', y='Quantity', data=df_monthly)
ax.set_title('Monthly Order Qty', fontdict={'family': 'PT Mono', 'size':16})


```




    Text(0.5, 1.0, 'Monthly Order Qty')




![png](/output/output_136_1.png)



```python
sns.set_style("whitegrid")
%config InlineBackend.figure_format='retina'
%matplotlib inline


# Check if sample sizes allow us to ignore assumptions;
# visualize sample size comparisons for two groups (normality check)
fig = plt.figure(figsize=(14,8))
ax = fig.gca()
ax = sns.barplot(x='OrderMonth', y='Quantity', data=df_monthly)
ax.set_title('Monthly Order Qty', fontdict={'family': 'PT Mono', 'size':16})


```




    Text(0.5, 1.0, 'Monthly Order Qty')




![png](/output/output_137_1.png)



```python
# Anova Test - Season + Quantity ()

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Quantity~C(OrderMonth)+Quantity:C(OrderMonth)", data=df_monthly).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
# reformat scientific notation of results for easier interpretation
anova_table.style.format("{:.5f}", subset=['PR(>F)'])
```




<style  type="text/css" >
</style><table id="T_6ac9e970_29bb_11ea_8476_f40f2405a054" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sum_sq</th>        <th class="col_heading level0 col1" >df</th>        <th class="col_heading level0 col2" >F</th>        <th class="col_heading level0 col3" >PR(>F)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_6ac9e970_29bb_11ea_8476_f40f2405a054level0_row0" class="row_heading level0 row0" >C(OrderMonth)</th>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row0_col0" class="data row0 col0" >7395.98</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row0_col1" class="data row0 col1" >11</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row0_col2" class="data row0 col2" >2.94204e+29</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row0_col3" class="data row0 col3" >0.00000</td>
            </tr>
            <tr>
                        <th id="T_6ac9e970_29bb_11ea_8476_f40f2405a054level0_row1" class="row_heading level0 row1" >Quantity:C(OrderMonth)</th>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row1_col0" class="data row1 col0" >772004</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row1_col1" class="data row1 col1" >12</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row1_col2" class="data row1 col2" >2.81504e+31</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row1_col3" class="data row1 col3" >0.00000</td>
            </tr>
            <tr>
                        <th id="T_6ac9e970_29bb_11ea_8476_f40f2405a054level0_row2" class="row_heading level0 row2" >Residual</th>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row2_col0" class="data row2 col0" >4.87009e-24</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row2_col1" class="data row2 col1" >2131</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row2_col2" class="data row2 col2" >nan</td>
                        <td id="T_6ac9e970_29bb_11ea_8476_f40f2405a054row2_col3" class="data row2 col3" >nan</td>
            </tr>
    </tbody></table>



### Normality

### Statistical


```python
# split orders into two groups (series): discount and fullprice order quantity
Jan = df_monthly.groupby('OrderMonth').get_group(1)['Quantity']

```


```python
# run tukey test for OQD (Order Quantity Discount) 
data = df_monthly['Quantity'].values
labels = df_monthly['OrderMonth'].values

import statsmodels.api as sms
model = sms.stats.multicomp.pairwise_tukeyhsd(data,labels)

# save OQD tukey test model results into dataframe (OQD: order quantity discount)
tukey_OQD = pd.DataFrame(data=model._results_table[1:], columns=model._results_table[0])
tukey_OQD
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1.3492</td>
      <td>0.9</td>
      <td>-4.6052</td>
      <td>7.3037</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>-1.8729</td>
      <td>0.9</td>
      <td>-7.4759</td>
      <td>3.73</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0.5014</td>
      <td>0.9</td>
      <td>-5.0704</td>
      <td>6.0733</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>-4.852</td>
      <td>0.358</td>
      <td>-11.2668</td>
      <td>1.5627</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>-3.2421</td>
      <td>0.9</td>
      <td>-11.428</td>
      <td>4.9438</td>
      <td>False</td>
    </tr>
    <tr>
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
      <td>61</td>
      <td>9</td>
      <td>11</td>
      <td>0.3585</td>
      <td>0.9</td>
      <td>-6.73</td>
      <td>7.4471</td>
      <td>False</td>
    </tr>
    <tr>
      <td>62</td>
      <td>9</td>
      <td>12</td>
      <td>2.2267</td>
      <td>0.9</td>
      <td>-4.4923</td>
      <td>8.9457</td>
      <td>False</td>
    </tr>
    <tr>
      <td>63</td>
      <td>10</td>
      <td>11</td>
      <td>-1.5082</td>
      <td>0.9</td>
      <td>-8.3215</td>
      <td>5.3051</td>
      <td>False</td>
    </tr>
    <tr>
      <td>64</td>
      <td>10</td>
      <td>12</td>
      <td>0.3599</td>
      <td>0.9</td>
      <td>-6.068</td>
      <td>6.7878</td>
      <td>False</td>
    </tr>
    <tr>
      <td>65</td>
      <td>11</td>
      <td>12</td>
      <td>1.8682</td>
      <td>0.9</td>
      <td>-4.8142</td>
      <td>8.5505</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 7 columns</p>
</div>



## Results

At a significance level of alpha = 0.05, we reject the null hypothesis which states there is no relationship between time of year (season) and sales revenue or volume of units sold. 

# Conclusion + Strategic Recommendations

- Conclusion & Strategic Recommendations
    1. 5% is the minimum discount level needed to produce maximum results. 
    - C: Offering discount levels < or > 5% either:
        a) has no effect on sales revenue and is therefore pointless
        b) increases loss in revenue despite higher order quantities that could have otherwise been achieved at only 5% discount (thereby maximizing revenue capture/minimizing loss).
    - R: Stop offering any discount other than 5%. 
    
    2. Continue to offer discounts in countries where they are effective in producing significantly higher order quantities. Stop offering discounts to countries where there is no effect on order quantities in order to minimize lost revenue. 
    
    3. Focus sales and marketing efforts in regions that produce highest revenue; consider  
    
    4.  


- Future Work
    * A. Gather and analyze critical missing data on customer types; investigate possible relationships between customer types and product categories (i.e. do certain customer types purchase certain 
    * C. Investigate possible relationship between regional revenues and shipping cost (i.e. is there a relationship between source (distributor) and destination (shipcountry) that might explain lower revenues in regions that are farther away in physical/geographic distance. 

# Future Work

Questions to explore in future analyses might include:

1. Build a product recommendation tool

2. Create discounts or free shipping offers to increase sales volumes past a certain threshold.
- Shipping Costs and Order Quantities/Sales Revenue
*Does shipping cost (freight) have a statistically significant effect on quantity? If so, at what level(s) of shipping cost?*

3. Customer Type and Product Category

*Is there a relationship between type of customer and certain product categories? If so, we can run more highly targeted sales and marketing programs for increasing sales of certain products to certain market segments.* 



# metricks

1. What were the top 3 selling products overall?
2. Top 3 selling products by country?
3. Top 3 selling products by region?
4. How did we do in sales for each product category?
5. Can we group customers into customer types (fill the empty database) and build a product recommendation tool?


```python
# Extract revenue per product category
cur.execute("""SELECT o.OrderId, o.CustomerId, od.ProductId, od.Quantity, od.UnitPrice, 
                od.Quantity*od.UnitPrice*(1-Discount) as Revenue, p.CategoryId, c.CategoryName
                FROM 'Order' o
                JOIN OrderDetail od 
                ON o.OrderId = od.OrderId
                JOIN Product p 
                ON od.ProductId = p.Id
                JOIN Category c
                ON p.CategoryId = c.Id
                ;""")
df = pd.DataFrame(cur.fetchall())
df.columns = [i[0] for i in cur.description]
print(len(df))
df.head(8)
```

    2155





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OrderId</th>
      <th>CustomerId</th>
      <th>ProductId</th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>Revenue</th>
      <th>CategoryId</th>
      <th>CategoryName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10248</td>
      <td>VINET</td>
      <td>11</td>
      <td>12</td>
      <td>14.0</td>
      <td>168.0</td>
      <td>4</td>
      <td>Dairy Products</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10248</td>
      <td>VINET</td>
      <td>42</td>
      <td>10</td>
      <td>9.8</td>
      <td>98.0</td>
      <td>5</td>
      <td>Grains/Cereals</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10248</td>
      <td>VINET</td>
      <td>72</td>
      <td>5</td>
      <td>34.8</td>
      <td>174.0</td>
      <td>4</td>
      <td>Dairy Products</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10249</td>
      <td>TOMSP</td>
      <td>14</td>
      <td>9</td>
      <td>18.6</td>
      <td>167.4</td>
      <td>7</td>
      <td>Produce</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10249</td>
      <td>TOMSP</td>
      <td>51</td>
      <td>40</td>
      <td>42.4</td>
      <td>1696.0</td>
      <td>7</td>
      <td>Produce</td>
    </tr>
    <tr>
      <td>5</td>
      <td>10250</td>
      <td>HANAR</td>
      <td>41</td>
      <td>10</td>
      <td>7.7</td>
      <td>77.0</td>
      <td>8</td>
      <td>Seafood</td>
    </tr>
    <tr>
      <td>6</td>
      <td>10250</td>
      <td>HANAR</td>
      <td>51</td>
      <td>35</td>
      <td>42.4</td>
      <td>1261.4</td>
      <td>7</td>
      <td>Produce</td>
    </tr>
    <tr>
      <td>7</td>
      <td>10250</td>
      <td>HANAR</td>
      <td>65</td>
      <td>15</td>
      <td>16.8</td>
      <td>214.2</td>
      <td>2</td>
      <td>Condiments</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group data by Category and get sum total revenue for each
df_category = df.groupby(['CategoryName'])['Revenue'].sum().reset_index()
df_category
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CategoryName</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Beverages</td>
      <td>267868.1800</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Condiments</td>
      <td>106047.0850</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Confections</td>
      <td>167357.2250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Dairy Products</td>
      <td>234507.2850</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Grains/Cereals</td>
      <td>95744.5875</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Meat/Poultry</td>
      <td>163022.3595</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Produce</td>
      <td>99984.5800</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Seafood</td>
      <td>131261.7375</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.CategoryId.value_counts()
```




    1    404
    4    366
    3    334
    8    330
    2    216
    5    196
    6    173
    7    136
    Name: CategoryId, dtype: int64




```python
# Explore sample sizes before testing
categories = df.groupby('CategoryName').groups
categories.keys()
```




    dict_keys(['Beverages', 'Condiments', 'Confections', 'Dairy Products', 'Grains/Cereals', 'Meat/Poultry', 'Produce', 'Seafood'])




```python
df_category.loc[df_category['CategoryName'] == 'Beverages']['Revenue'].sum()
```




    267868.17999999993




```python
#create dict of months and order quantity totals
rev_per_cat = {}

for k,v in categories.items():
    rev = df_category.loc[df_category['CategoryName'] == k]['Revenue'].sum()
    rev_per_cat[k] = rev

rev_per_cat
```




    {'Beverages': 267868.17999999993,
     'Condiments': 106047.08500000002,
     'Confections': 167357.22499999995,
     'Dairy Products': 234507.285,
     'Grains/Cereals': 95744.58750000001,
     'Meat/Poultry': 163022.3595,
     'Produce': 99984.57999999999,
     'Seafood': 131261.73750000002}




```python
# plot order quantity totals by month
fig = plt.figure(figsize=(12,12))
for k,v in rev_per_cat.items():
    plt.bar(x=k, height=v)
```


![png](/output/output_156_0.png)



```python
# What were the top 3 selling product categories in each region or country?
# What were the lowest 3 selling product categories in each region or country?
```
