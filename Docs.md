# Open Repository for Data Sciences - Documentation

The purpose of this notebook is to provide more detail surrounding the functions that are implemented in the Open Repository for Data Sciences learning tool. This document also gives the user an introduction to basic machine learning (ML) terminology and concepts that are fundamental to practicing data science.

Many of these functions utilize popular python data science libraries like `pandas`, `NumPy`, `matplotlib`, `Seaborn`, and `scikit-learn` to name a few.

* `pandas` - Pandas is a package for data manipulation and analysis. It offers flexible operations for manipulating tabular and time series data.

* `NumPy` - Numpy is a package for scientific computing which provides support for performing computations with multi-dimensional arrays and matrices. It also ships with a number of high-level mathematical functions to operate on these arrays.

* `matplotlib` - Matplotlib is python package for creating plots. `seaborn` came out later which was built on top of Maplotlib. Seaborn improved upon Matplotlib by developing a more concise syntax and integrating more attractive and informative statistical graphics. Seaborn is built to operate on dataframes and arrays.

* `scikit-learn` - Scikit-learn is an open-source python libarary for implementing ML algorithms. It features many popular choices for implementing classification, regression, and clustering algorithms.

These descriptions may in fact be a gross simplification of what you can do with these frameworks. However, this is a good starting point, especially if this is your first exposure to these tools. 


```python
from data import load_dataframe, names
```

## List of named datasets

The learning tool ships with several different types of datasets. These are their names within the coding infrastructure. It is important to know their names if you want to be able to execute these functions yourself or modify existing functions in the source code.


```python
names
```


    ['exam_scores', 'petroleum_consumption', 'study_hours', 'wine_quality']

### pandas DataFrames are the core data structure being operated on behind the scenes

The raw data files are stored within the project's `/data` directory. To transform the text file into an object we can operate on, the helper function `load_dataframe` is used. 

It is important to note that this function takes as input a **pandas DataFrame (df)** 

The function is called like so: `load_dataframe(df)`

### Example of using `load_dataframe`


```python
from IPython.display import display
```


```python
load_dataframe('exam_scores')
```

    Creating pandas DataFrame for exam_scores dataset...

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Exam_1</th>
      <th>Exam_2</th>
      <th>Admitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.623660</td>
      <td>78.024693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.286711</td>
      <td>43.894998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.847409</td>
      <td>72.902198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.182599</td>
      <td>86.308552</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.032736</td>
      <td>75.344376</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>83.489163</td>
      <td>48.380286</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>42.261701</td>
      <td>87.103851</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>99.315009</td>
      <td>68.775409</td>
      <td>1</td>
    </tr>
    <tr>
      <th>98</th>
      <td>55.340018</td>
      <td>64.931938</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>74.775893</td>
      <td>89.529813</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
## Let's take a peek at all the datasets


```python
for n in names:
    display(load_dataframe(n).head())
    print()
```

    Creating pandas DataFrame for exam_scores dataset...

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Exam_1</th>
      <th>Exam_2</th>
      <th>Admitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.623660</td>
      <td>78.024693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.286711</td>
      <td>43.894998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.847409</td>
      <td>72.902198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.182599</td>
      <td>86.308552</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.032736</td>
      <td>75.344376</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
    Creating pandas DataFrame for petroleum_consumption dataset...

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Petrol_Tax</th>
      <th>Average_Income</th>
      <th>Paved_Highways</th>
      <th>Population_Driver_Licence_Perc</th>
      <th>Petrol_Consumption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.0</td>
      <td>3571</td>
      <td>1976</td>
      <td>0.525</td>
      <td>541</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>4092</td>
      <td>1250</td>
      <td>0.572</td>
      <td>524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>3865</td>
      <td>1586</td>
      <td>0.580</td>
      <td>561</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.5</td>
      <td>4870</td>
      <td>2351</td>
      <td>0.529</td>
      <td>414</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>4399</td>
      <td>431</td>
      <td>0.544</td>
      <td>410</td>
    </tr>
  </tbody>
</table>
    Creating pandas DataFrame for study_hours dataset...  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
  </tbody>
</table>

​    
    Creating pandas DataFrame for wine_quality dataset...

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>

​    


### Let's store these pandas DataFrames so we can re-use them later in the docs


```python
data = {n: load_dataframe(n) for n in names}
```

    Creating pandas DataFrame for exam_scores dataset...
    Creating pandas DataFrame for petroleum_consumption dataset...
    Creating pandas DataFrame for study_hours dataset...
    Creating pandas DataFrame for wine_quality dataset...


### Now we can access any DataFrame we want by providing its corresponding name


```python
data['study_hours']
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.5</td>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.3</td>
      <td>81</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.7</td>
      <td>25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.7</td>
      <td>85</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5.9</td>
      <td>62</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4.5</td>
      <td>41</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.3</td>
      <td>42</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.9</td>
      <td>95</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.9</td>
      <td>24</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6.1</td>
      <td>67</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.4</td>
      <td>69</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.7</td>
      <td>30</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4.8</td>
      <td>54</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.8</td>
      <td>35</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6.9</td>
      <td>76</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7.8</td>
      <td>86</td>
    </tr>
  </tbody>
</table>


## Exploratory Data Analysis

There are three functions which come with this module. The main one you will see implemented in the learning tool is the `visualize_corr_matrix`. This outputs a pretty heatmap of the correlations between features within the dataset. This is a useful tool and can be helpful in indentifying cases of **multicollinearity** when performing multiple linear regression. 

> ### What is Multicollinearity?

> Multicollinearity occurs when independent variables in a regression model are correlated. This correlation is a problem because independent variables should be independent. If the degree of correlation between variables is high enough, it can cause problems when you fit the model and interpret the results.

> A key goal of regression analysis is to isolate the relationship between each independent variable and the dependent variable. The interpretation of a regression coefficient is that it represents the mean change in the dependent variable for each 1 unit change in an independent variable when you hold all of the other independent variables constant. That last portion is crucial for our discussion about multicollinearity.

> The idea is that you can change the value of one independent variable and not the others. However, when independent variables are correlated, it indicates that changes in one variable are associated with shifts in another variable. The stronger the correlation, the more difficult it is to change one variable without changing another. It becomes difficult for the model to estimate the relationship between each independent variable and the dependent variable independently because the independent variables tend to change in unison.

> Taken from [Statistics By Jim](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)


```python
from exploratory_data_analysis import get_descriptive_statistics, visualize_corr_matrix, get_correlated_features
```

### Example using `get_descriptive_statistics`


```python
get_descriptive_statistics(data['petroleum_consumption'])
```

    Dataset Size:
    
    There are 48 rows and 5 columns in the dataset.
    
    ---------------------------------------------------
    
    Column Names and Types:
    
    Petrol_Tax                        float64
    Average_Income                      int64
    Paved_Highways                      int64
    Population_Driver_Licence_Perc    float64
    Petrol_Consumption                  int64
    
    ---------------------------------------------------
    
    Basic Statistics:
    
    Petrol_Tax
    
    count    48.000000
    mean      7.668333
    std       0.950770
    min       5.000000
    25%       7.000000
    50%       7.500000
    75%       8.125000
    max      10.000000
    
    Average_Income
    
    count      48.000000
    mean     4241.833333
    std       573.623768
    min      3063.000000
    25%      3739.000000
    50%      4298.000000
    75%      4578.750000
    max      5342.000000
    
    Paved_Highways
    
    count       48.000000
    mean      5565.416667
    std       3491.507166
    min        431.000000
    25%       3110.250000
    50%       4735.500000
    75%       7156.000000
    max      17782.000000
    
    Population_Driver_Licence_Perc
    
    count    48.000000
    mean      0.570333
    std       0.055470
    min       0.451000
    25%       0.529750
    50%       0.564500
    75%       0.595250
    max       0.724000
    
    Petrol_Consumption
    
    count     48.000000
    mean     576.770833
    std      111.885816
    min      344.000000
    25%      509.500000
    50%      568.500000
    75%      632.750000
    max      968.000000



### Example using `visualize_corr_matrix`


```python
visualize_corr_matrix(data['wine_quality'])
```


![png](output_21_0.png)


### Example using `get_correlated_features`

This function also takes as input a `target`. This is what we are trying to **predict**. 

The function call `get_correlated_features(df, target)` will return features (attributes) that are correlated with the target feature. Internally, a **threshold** of 0.2 is used on the Pearson Correlation Coefficent (r); although, you can control this with the `threshold` parameter i.e. `get_correlated_features(df, target, threshold=0.4)`


```python
get_correlated_features(data['wine_quality'], target='quality')
```

    Correlations with quality:
    
    fixed acidity           0.124052
    volatile acidity       -0.390558
    citric acid             0.226373
    residual sugar          0.013732
    chlorides              -0.128907
    free sulfur dioxide    -0.050656
    total sulfur dioxide   -0.185100
    density                -0.174919
    pH                     -0.057731
    sulphates               0.251397
    alcohol                 0.476166
    
    Optimal features based on absolute threshold: 0.2
    
    volatile acidity    0.390558
    citric acid         0.226373
    sulphates           0.251397
    alcohol             0.476166


# Algorithm Implementations


```python
from algorithms import run, implementations
```

There are four main types of machine-learing algorithms that are implemented. Below are the following **named** implementations.


```python
implementations
```


    ['Simple Linear Regression',
     'Naive Bayes',
     'Random Forest',
     'Support Vector Machines']



## Simple Linear Regression

In the Simple Linear Regression implementation an artificial two-dimensional dataset is generated to nicely visualize the predictions of the model. 


```python
run('Simple Linear Regression')
```

    Building Modeling...
    Fitting model to data...
    Generating Predictions...




![png](output_30_1.png)



![png](output_30_2.png)

## Naive Bayes

A variant of Naive Bayes, specifically Multinomial Naive Bayes is demonstrated to classify Amazon Reviews as having either positive (1) or negative (0) sentiment.

### Data Cleaning

The first step in this workflow is to clean the data. This is a typical first step (after the data is collected) that a Data Scientist will need to do. This can often be one of the most time-consuming phases. The reason for this is that data is often messy and unstructured. In order to apply fancy data science techniques, we need the data to be in a nice format and arriving at this can involve many tedious procedures.

### Data Preprocessing 

Data can be noisy. When dealing with text data it is frequently the case that we want to remove commonly used words. The intuition behind this is that commonly used words are uninformative and an abundance of these words can create "noise" around the true "pattern" or "signal" we are trying to extract from the data. In this example, we are trying to extract the pattern of language which corresponds to positive vs. negative sentiment. 

### Understanding Trends in Generalization Performance

A key theme you will see again and again when practicing data science is "more data the better". Although, not always the case! We need high quality data as well!

The Naive Bayes implementation comes with a nice graphic that allows you to visualize this concept in action.


```python
run('Naive Bayes')
```

    Preview of Amazon Reviews Dataset (Uncleaned): 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>So there is no way for me to plug it in here i...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Good case</td>
      <td>Excellent value.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great for the jawbone.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tied to charger for conversations lasting more...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The mic is great.</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I have to jiggle the plug to get it to line up...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>If you have several dozen or several hundred c...</td>
      <td>then imagine the fun of sending each of them ...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>If you are Razr owner...you must have this!</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Needless to say</td>
      <td>I wasted my money.</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What a waste of money and time!.</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

    Cleaning Dataset...
    
    Preview of Amazon Reviews Dataset (After Data Cleaning Phase):

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>so there is no way for me to plug it in here i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>good case excellent value</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great for the jawbone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tied to charger for conversations lasting more...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>the mic is great</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>i have to jiggle the plug to get it to line up...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>if you have several dozen or several hundred c...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>if you are razr owneryou must have this</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>needless to say i wasted my money</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>what a waste of money and time</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
    Data Preprocessing...
    Removed 107 stop words.
    
    Multinomial Naive Bayes Accuracy: 0.928



![png](output_33_5.png)


## Random Forest

In the Random Forest implementation we generated another two-dimensional toy dataset. However this time the red and blue classes are NOT linearly separable.

A Random Forest is a collection of many machine learning classifiers known as Decision Trees or just trees for short. You can view each tree as a series of "splits" on each feature in order to accurately classify the data points. For example, a potential split to categorize the blue and red data points might be a split on Feature 1 at 0.25 and then another on Feature 0 at -0.25 and so on. These tree are designed to terminate splitting once all of the data points have been correctly classified. Because of this, these types of algorithms are naturally prone to becoming overly complex and as a result will fail to generalize well to unseen data. What this means is that they have essentially memorized in such granular detail the data of which it was trained on the miss the pattern that exists in the general population of data points. This term in machine learning is known as **overfitting** To combat this, randomness is injected into the creation of each tree. The final model is an aggregation of these randmoized trees which we call the **Random Forest**

This implementation displays a few of the unique, randomized trees within the ensemble of trees and also displays the final decision boundary/classifications of the Random Forest.


```python
run('Random Forest')
```

    Accuracy on training set: 0.99
    Accuracy on test set: 0.96



![png](output_36_1.png)



![png](output_36_2.png)



![png](output_36_3.png)


## Support Vector Machines

The last type of ML implementation is called Support Vector Machines. The idea behind this approach is that we may not always have data that is linearly separable but it may be linearly seperable in a different feature representation, typically a higher-dimensional space. This is commonly referred to as the "kernel trick", that is, using a linear classifier to solve a non-linear problem. The Kernel function is what we used to map the original data points into a higher-dimensional space in which they become linearly separable. There a few different types but here we just create another toy two-dimensional dataset and display the output of each method that is implemented in the `scikit-learn` package


```python
run('Support Vector Machines')
```


![png](output_39_0.png)



![png](output_39_1.png)



![png](output_39_2.png)

