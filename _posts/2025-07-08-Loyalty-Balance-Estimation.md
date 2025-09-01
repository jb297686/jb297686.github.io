---
layout: post
title: “Loyalty Matters” – Predicting Loyalty Point Balances Using Machine Learning
image: "/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

Here at *We-Do-Data-Science-So-You-Don't-Have-To Limited* our work is never done! This morning a (not very) well-known supermarket chain got in touch to see if we could use our machine learning skills to re-create some important customer data that got deleted in a data centre incident. Let's get to it!

---

# Table of Contents
- [1. Project Overview](#project-overview)
  - [Context](#project-overview-context)
  - [Actions](#project-overview-actions)
  - [Results](#project-overview-results)
  - [Growth/Next Steps](#project-overview-growth-next-steps)
- [2. Data Overview](#data-overview)
- [3. Modelling Approach](#modelling-approach)
- [4. Linear Regression](#linear-regression)
- [5. Decision Tree](#decision-tree)
- [6. Random Forest](#random-forest)
- [7. Modelling Summary](#modelling-summary)
- [8. Estimating Customer Points Balances](#modelling-predictions)
- [9. Growth and Next Steps](#growth-next-steps)

---

# Project Overview <a name="project-overview"></a>
### Context <a name="project-overview-context"></a>
The supermarket management are in a bit of a pickle following the incident at their data centre. Some key customer data got wiped from their corporate database in what we've been told was a 'fat finger' incident. Apparently the CFO arrived at the data centre unannounced in the early hours of the morning and told the night shift IT technician that he needed to do an ad-hoc audit of the accounting information in the corporate database. Somehow, while running his audit procedure, the CFO managed to accidentally delete data from two key database tables. One table contained details of all senior management hospitality account expenditure over the last three years. We've been told not to worry about that and to focus our efforts on the other affected table, which contained the loyalty card points balances of all the supermarket's customers.

The senior execs fear that some customers will be very annoyed if they find their loyalty card balances have been wiped-out. The CEO is in a panic and has suggested giving each customer a very generous balance of 15,000 points. However, such a blunt approach would cost the company a lot of money (in-line with the industry norm, every 1,000 points can be exchanged for 20p off a fruit scone when spending over £13 in the in-store cafes before 9am on the Wednesday after a full moon, excluding stores in Wales). This approach would also leave those customers who had balances in excess of 15,000 points feeling short-changed. Can we come up with a more sophisticated remedy?

Fortunately, the point of sale system keeps a record of the points balance of each customer who made a purchase during the last week, so we have the points balances of about half the loyalty card holders, and the client is also giving us access to some database tables that contain information they think we may be able to make use of. Let's see if we can use machine learning techniques to come up with a reliable way of estimating loyalty card balances. We need to move quickly, before word of the missing balances hits social media!

### Actions <a name="project-overview-actions"></a>
We'll start by compiling customer metrics (input variables) that may help us estimate the output variable (loyalty card points balance) and by splitting the customers into two groups: those for whom the points balance is known (from the point of sale system) and those for whom we need to estimate it.

We want to make estimates of a numeric output metric, so we will try three approaches:
- Linear Regression
- Decision Tree
- Random Forest

### Results <a name="project-overview-results"></a>
Our testing found that the random forest approach had the higest predictive accuracy.

##### Metric 1: Adjusted R-Squared (based on test set)
- Random Forest: 0.952
- Decision Tree: 0.898
- Linear Regression: 0.769

##### Metric 2: R-Squared using K-Fold Cross-Validation (with k=4)
- Random Forest: 0.925
- Linear Regression: 0.847
- Decision Tree: 0.840

For this project accuracy was the most important consideration, so we chose the random forest even though the other two models can sometimes be more straightforward to explain to client stakeholders.

In the future our model will be useful for the client when analysing locations for new stores and for tailoring loyalty points offers to customers.

### Growth / Next Steps <a name="project-overview-growth-next-steps"></a>
While our predictive accuracy was relatively high (95.2% based on our test set with our selected model) we could look to further improve accuracy by tuning the random forest hyperparameters such as maximum tree depth and the number of trees in the forest.

In terms of data, we could ask our client if their loyalty points have an expiration date and if they have any data that shows the percentage of loyalty card points that typically expire without being used. We could use that data to further refine our points balance estimates.

---

# Data Overview  <a name="data-overview"></a>
Our client has provided us with copies of three database tables. We need to estimate the *points_balance* metric for the 50% of customers whose points balance is missing. The *points_balance* metric exists in the *customer_loyalty* table.  The *customer_details* table contains a list of all the customers who have loyalty cards.

We hypothesise that we can use metrics from the *transactions* table and *customer_details* table to predict (estimate) the missing points balances. Using pandas we'll merge these tables together to create a single dataset for modelling which we'll then save out using pickle.

```python
# Import required packages
import pandas as pd
import pickle

# Import the data
customer_loyalty = pd.read_excel("data/grocery_data_from_client.xlsx", sheet_name="customer_loyalty")
transactions = pd.read_excel("data/grocery_data_from_client.xlsx", sheet_name="transactions")
customer_details = pd.read_excel("data/grocery_data_from_client.xlsx", sheet_name="customer_details")

# Add the existing points balance data onto the main customer dataset
data_for_regression = pd.merge(customer_details, customer_loyalty, how="left", on="customer_id")

# Aggregate transaction data by customer
sales_summary = transactions.groupby("customer_id").agg({"sales_cost": "sum",
                                                         "num_items": "sum",
                                                         "transaction_id": "count",
                                                         "product_area_id": "nunique"}).reset_index()

# Make column names clearer
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]

# Create an average basket column for each customer
sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

# Merge the transaction summary data into the main dataset
data_for_regression = pd.merge(data_for_regression, sales_summary, how="inner", on="customer_id")

# Drop the customer_id field as it's just a unique ID so shouldn't have predictive power
data_for_regression.drop(["customer_id"], axis=1, inplace=True)

# Split data into modelling (known points balance) and inference (unknown points balance) datasets
modelling_data = data_for_regression.loc[data_for_regression["points_balance"].notna()]
data_for_inference = data_for_regression.loc[data_for_regression["points_balance"].isna()]

# Drop the column with unknown points balances from the data for inference
data_for_inference.drop(["points_balance"], axis=1, inplace=True)

# Save data to file for future use
pickle.dump(modelling_data, open("data/regression_modelling_data.p", "wb"))
pickle.dump(data_for_inference, open("data/regression_data_for_inference.p", "wb"))
```
Our dataset now contains the following fields:

|**Variable name**|**Type**|**Description**|
|---|---|---|
| points_balance | Output | The customer's loyalty card points balance |
| distance_from_store | Input | The distance (in km) that the customer lives from their nearest branch |
| credit_score | Input | A value in range 0 to 1 indicating customer creditworthiness, as assessed by an independent credit rating agency |
| gender | Input | The gender provided by the customer |
| transaction_count | Input | Total number of transactions the customer made in the supermarket in the last year |
| total_sales | Input | Total amount (in £) the customer spent in the supermarket in the last year |
| average_basket_value | Input | Customer's average spend (in £) per transaction at the supermarket in the last year |
| total_items | Input | Total number of items the customer purchased in the supermarket in the last year |
| product_area_count | Input | Number of the supermarket's product areas customer purchased items from in the last year |   

---

# Modelling Approach  <a name="modelling-approach"></a>
We'll model the relationship between *points_balance* and the input variables for the customers that we have points balances for, retaining a subset of the data for testing. If our testing shows that we are able to reliably estimate *points_balance* based on the input variables, we'll then proceed to make estimates of the points balances for customers whose *points_balance* data was wiped by the CFO.

The three types of model that we will train are:
- Linear Regression
- Decision Tree
- Random Forest

---

# Linear Regression  <a name="linear-regression"></a>
We'll use the scikit-learn library in Python to create our models. The code to do this is below, broken down into four sections:
- Data import
- Data preprocessing
- Model training
- Performance assessment

### Data Import
Firstly we load the data that we saved in pickle format above and shuffle it to ensure we are not biasing the split into training and test subsets. When shuffling we pass a value for the *random_state* parameter. This is just to make the shuffling reproducable, which is useful if we want to try different values of model hyperparameters and make fair comparisons between the different hyperparemeter settings.

```python
# Import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler

# Import the pickle file we saved earlier
data_for_model = pickle.load(open("data/regression_modelling_data.p", "rb"))
data_for_model = shuffle(data_for_model, random_state=42)
```

### Data Preprocessing
For regression analysis we need to address:
- Missing values in the data
- The effect of outliers
- Encoding categorical variables to numeric form and avoiding multicollinearity
- Feature selection

##### Mising Values
It turns out that only a small handful of rows having missing values, so instead of trying to impute values for them we'll just remove them.

```python
# Remove rows that have missing values
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```

##### Dealing With Outliers
In regression analysis outlying data points can affect the model's ability to generalise well to new data. However, first we should carefullly consider whether the outliers are genuinely exceptional values. Let's create box a box plot for each variable to see if this applies to any of our data.

```python
potential_outlier_vars = ["points_balance", "distance_from_store", "transaction_count", "total_sales", "average_basket_value", "total_items"]

for model_var in potential_outlier_vars:
    data_for_model[model_var].plot(kind="box", vert=False)
    plt.title(f"Box Plot Showing Spread of Data for {model_var}")
    plt.show()

for column in ["points_balance", "distance_from_store"]:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
```

![Box plot for points_balance](/img/posts/loyalty_regresion_box_plot_points_balance.png)
<br>
<br>
![Box plot for distance_from_store](/img/posts/loyalty_regresion_box_plot_distance_from_store.png)

The box plot for *points_balance* shows that two customers have balances in excess of 100,000. To find out if that's realistic or not, we managed to open a line of communication with the IT manager at the supermarket's data centre, who revealed that:
1. The supermarket ran a competition last year and the winner was awarded 150,000 loyalty points. There are no plans to hold any more competitions going forward.
2. A few months back the CFO asked for his own points balance to be increased to 130,000 as he wanted to know if the database was "strong enough to handle big numbers". Once he'd concluded that it was, nobody knew whether they were supposed to revert the CFO's points balance to its previous value, and so in the end nobody did.
3. The *distance_from_store* values were captured by asking each customer how far they lived from their nearest store and, since some people are bad at estimating distances, any outlying values are more than likely unreliable.

In light of this information we decided to remove the rows of data with outlying values in the *points_balance* and *distance_from_store* fields. In total four rows of data were removed from our modelling dataset.

```python
data_for_model.drop(outliers, inplace=True)
```

##### Splitting Datasets for Modelling
Next, we split the input variable out into its own DataFrame and then we use the classic 80-20 split to create the training set (which we'll use to train the model) and test set (which we'll use to evaluate model performance).

```python
# Split the data into input variables (X) and output variable (y)
X = data_for_model.drop(["points_balance"], axis=1)
y = data_for_model["points_balance"]

# Allocate 80% of data for model training leaving 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

##### Categorical Input Variables
In the data provided by the client there is one categorical variable, *gender*, which can take either 'M' or 'F' as its value. For regression analysis we need to convert this to a numerical value. By default the Pandas one-hot encoder would encode the categorical variable as two columns but we drop one of them to avoid dummy variable multicollinearity (where value of one column perfectly predicts another column's value) which can make it difficult to interpret the regression coefficients.

Note that we fit the encoder (that is, we ask it to learn a suitable encoding method for our data field) on the training set *only*. This is a good practice, as if we were to ask an encoder to determine the appropriate encoding method by inspecting both the training *and* test sets when we then applied the encoder's transform to the training set we would be 'leaking' some information about the test set into the training set, thereby invalidating our assessment of our model's performance using the test set. While this is unlikely to be an issue in this particular case, it's a good habit to be in.

```python
# List of categorical variables that need encoding
categorical_vars = ["gender"]

# Create the encode, drop="first" to avoid multicollinearity
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Encode the categorical variable
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the names of the encoded features
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Replace the categorical variable with the encoded equivalent
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_vars, axis=1, inplace=True)
```

Having run the above code our *gender* variable has been replaced by a new variable *gender_M*.

##### Feature Selection
We have eight input variables at our disposal, but it may not make sense to use them all in our regression model. Reducing the number of variables we use has the following benefits:
- **Reduced computational expense**: complex models can take a long time to train
- **Improved accuracy**: removing variables that don't play a part in predicting the output variable reduces noise and improves model accuracy
- **Improved stakeholder buy-in**: some managers at our supermarket client may be sceptical about the impact of some of the input variables, so we'll want to be able to demonstrate where a relationship really does exist between an input variable and the output variable

We'll start using a simple form of analysis that will help us build an understanding of the importance of each input variable. The table below shows each input variable's correlation with *points_balance* based on the training data.

```python
# Create with a correlation matrix
data_for_model_with_encodings = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
correlation_matrix = data_for_model_with_encodings.corr()
```

|**Variable**|**Correlation**|
|---|---|
|distance_from_store|-0.74|
|gender_M|0.62|
|transaction_count|0.55|
|total_items|0.37|
|product_area_count|0.35|
|total_sales|0.33|
|average_basket_value|0.17|
|credit_score|-0.01|

We can see that:
- *distance_from_store* is negatively correlated with *points_balance*, which makes intuitive sense - customers who live further away from the supermarket's branches probably shop there less often than those who live closer
- *credit_score* has very low correlation with *points_balance* suggesting that a customer's creditworthiness is not a good predictor of their tendency to collect supermarket loyalty points

Next, let's use a more sophisticated feature selection approach called *Recursive Feature Elimination with Cross Validation* (RFECV), which assesses how well our variables, all considered together, can predict the points balances. This method starts with all the input variables and iterively removes those that are found to have the weakest relationship with the output variable. It does this on different portions ('folds') of the training data and then takes the average performance across the folds to determine the optimal number of variables to include.

```python
# Instantiate the regression model and the itertive feature selector
regressor = LinearRegression()
feature_selector = RFECV(regressor)

# Normalize the data
normalizer = MinMaxScaler()
X_train_RFECV = pd.DataFrame(normalizer.fit_transform(X_train), columns=X_train.columns)
y_train_RFECV = pd.DataFrame(normalizer.fit_transform(pd.DataFrame(y_train)), columns=[y_train.name])
                        
# Fit the feature selector to our training data
fit = feature_selector.fit(X_train_RFECV, y_train_RFECV)

# Print the optimal number of features, as determined by the feature selector
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")
print(f"Features to drop: {X_train.columns[~feature_selector.get_support()]}")
```

We then visualize the RFECV results in a plot.

```python
# Plot results using each possible number of input variables
plt.plot(fit.cv_results_["n_features"], fit.cv_results_["mean_test_score"], marker="o")
plt.ylabel("Model score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFECV \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_["mean_test_score"]),4)})")
plt.tight_layout()
plt.show()
```
![Recursive feature evaluation with cross-validation](/img/posts/loyalty_regresion_RFECV.png)

The RFECV analysis suggests that the optimal approach is to use six of our eight input variables, dropping *credit_score* and *gender_M*. For *credit_score* this is not surprising given its individual correlation with *points_balance*. For *gender_M* the picture is a bit different: its correlation with *points_balance* is resonably high (0.62) so we might expect it to be important in predicting points balance. However, there are a couple of points we need to remember:
1. The correlation coefficient of 0.62 shows us that *points_balance* and *gender_M* tend to move in the same direction. However, the correlation coefficient only tells us about how *often* two variables move in the same direction, and does not tell us how *much* one variable moves in response to a change in the other variable. That is, it tells us only about *direction* of movement, and not about *magnitude* of movement, and for a regression analysis the latter is important.
2. Another possibility is that *gender_M* is highly correlated with one of other input variables and therefore its inclusion doesn't add much to the analysis. However, we inspected the correlation matrix for the input variables and found that *gender_M* is not highly correlated with any of the other input variables.
3. The correlation of an individual input variable does not take account of interactions between the variables - which is what we hope to capture in a multi-variate regression analysis. RFECV tells us the variable's contribution to prediction, not just its individual assocation with the output variable.

We'll use the RFECV analysis to remove the two input variables that do not have predictive power for our output variable.

```python
# Reduce the number of variables to the optimal level
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]
```

Right, on we go!

### Training the Model
We're now ready to train our linear regression model.

```python
# Instantiate the regression model
regressor = LinearRegression()
# Fit the model to our training input and output variable data
regressor.fit(X_train, y_train)
```

### Model Performance
##### Make Predictions on the Test Set
To see how good our model is at estimating the missing points balance we use it to predict the *points_balance* values for our test set.

```python
# Predict points_balance values for the test set
y_pred = regressor.predict(X_test)
```

##### Calculate R-Squared
The R-Squared value shows the proportion of variation in the output variable (*points_balance*) that is explained by the input variables. In other words, if we think of the total variation of the output variable values around the mean points balance, R-Squared tells us how much of the variation is explained by our model, and how much is explained by other, unknown factors.

We can calculate R-Squared using our predicted points balances for the test set and the accompanying known test set points balances.

```python
# Calculate R-Squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

This gives an R-Squared of 0.787, indicating that our model explains 78.7% of the variation in points balances in the test set.

##### Calculate R-Squared via Cross Validation
We can go one step further and ask scikit-learn to calculate R-Squared via a process of 'cross-validation', which involves splitting the training set into chunks (we'll specify four chunks), and using each chunk in turn as a test set for a regression model trained on the other chunks, then calculating the average R-Squared from each such run.

```python
# Cross validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
print(cv_scores.mean())
```
The cross-validated R-Squared for our model is 0.847.

##### Calculated Adjusted R-Squared
With linear regression, adding additional input variables into the mix can only ever increase R-Squared, and never decrease it, so with multi-variate linear regression it's important to adjust R-Squared such that an additional variable only makes a positive contribution to R-Squared if its contribution to explaining the variation in the output variable is above that which could be expected by random chance.

```python
# Calculate Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```
This gives an adjusted R-Squared of 0.769, which is slightly lower than the unadjusted R-Squared value of 0.787.

### Model Summary Statistics
We can extract the model coefficients and intercept.

```python
# Extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis=1)
summary_stats.columns = ["input_variable", "coefficient"]

# Extract model intercept
regressor.intercept_
```

|**Variable**|**Coefficient**|
|---|---|
|intercept|3495.3|
|distance_from_store|-2729.7|
|product_area_count|856.8|
|average_basket_value|105.9|
|transaction_count|28.2|
|total_items|10.2|
|total_sales|-1.24|

Not surprisingly, the coefficient for *distance_from_store* is negative, as customers who live farther away probably visit the store less often.

What did surprise us though was that the coefficient of *total_sales* was negative, as we might expect higher spending to translate to higher points balances. We discussed this with our client who explained that this supermarket's loyalty scheme is aimed at driving stock turnover and so gives customers more points the more items they purchase (rather than the more money they spend). A customer who buys several inexpensive items would earn more loyalty points than one who buys a small number of expensive items, which explains the negative coefficient of *total_sales*. The scheme also sometimes gives customers large bonus point awards for buying products in different categories, hence the positive relationship between *points_balance* and *product_area_count*.

---

# Decision Tree <a name="decision-tree"></a>
Next, we'll see if we can get more accurate predictions using the scikit-learn library in Python to create a decision tree. The code to do this is below, broken down into four sections:
- Data import
- Data preprocessing
- Model training
- Performance assessment

### Data Import
We again load the data that we saved in pickle format and shuffle it.
```python
# Import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# Import the pickle file we saved earlier and shuffle it
data_for_model = pickle.load(open("data/regression_modelling_data.p", "rb"))
data_for_model = shuffle(data_for_model, random_state=42)
```

### Data Preprocessing
##### Missing Values
Next, just as we did for linear regression, we remove the small number of data points that have missing values.

```python
# Remove rows that have missing values
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```

##### Dealing With Outliers
Decision tree models are generally pretty robust against the effects of outliers. However, we know that we have two *extreme* outliers in our *points_balance* data (the balances of the CFO and the competition winner). If those two data points (by chance) end up in our test set, the predictions of the model we create using our training set will likely make very inaccurate predictions for those two data points, which would mean our accuracy scores would not give us a true reflection of our model's predictive power. So we'll remove the two extreme outliers.

Since our client is sure that there are only these two extreme value of *points_balance* we can remove the two outliers without having to worry that our model *should* be predicting extreme points balances for some of the missing balances.

```python
potential_outlier_vars = ["points_balance"]

for column in ["points_balance"]:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace=True)
```

##### Splitting Datasets for Modelling
We split the input variable out into its own DataFrame use an 80-20 split for the training and test data.
```python
# Split the data into input variables (X) and output variable (y)
X = data_for_model.drop(["points_balance"], axis=1)
y = data_for_model["points_balance"]

# Allocate 80% of data for model training leaving 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

##### Categorical Input Variables
Just like linear regression, the decision tree model cannot deal with data in categorical variables with string (here 'M' or 'F') categories, so we use one-hot encoding to turn our encode our categorical variable.

```python
# List of categorical variables that need encoding
categorical_vars = ["gender"]

# Create the encode, drop="first" to avoid multicollinearity
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Encode the categorical variable
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the names of the encoded features
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Replace the categorical variable with the encoded equivalent
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_vars, axis=1, inplace=True)
```

### Training the Model
Next we instantiate our decision tree model and train it using our training set. 
```python
# Instantiate the decision tree model
regressor = DecisionTreeRegressor(random_state=42)
# Fit the model to our training input and output variable data
regressor.fit(X_train, y_train)
```

### Model Performance
##### Make Predictions on the Test Set
```python
# Predict on the test set
y_pred = regressor.predict(X_test)
```

##### Calculate R-Squared
We calculate R-Squared using our predicted points balances for the test set (*y_pred*) and the accompanying known test set points balances (*y_test*).
```python
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```
The R-Squared score is 0.908.

##### Calculate R-Squared via Cross Validation
As for linear regression we'll use cross-validation to find the average accuracy score using four folds.
```python
# Cross validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
print(cv_scores.mean())
```
The mean cross-validated R-Squared value is 0.840, which is very slightly lower than the 0.847 we saw for cross-validated linear regression accuracy.

##### Calculated Adjusted R-Squared
We also calculate adjusted R-Squared, which ensures that an additional variable only makes a positive contribution to the accuracy measure if its contribution to explaining the variation in the output variable is above that which could be expected by random chance.

```python
# Calculate Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The adjusted R-Squared is 0.898, which, as expected, is slightly lower than the unadjusted R-Squared.

### Decision Tree Regularization
If we don't limit the depth of a decision tree it will continue splitting nodes until it perfectly predicts each data point in the training set. Overfitting the training data is undesirable as it will limit the model's ability to generalise to new data, so we will need to limit the depth the tree can extend to in order to prevent this overfitting. However, if we limit the depth of the tree too much we'll find that it struggles to make accurate predictions.

To find the optimal depth we'll test different values of maximum depth and create a plot which will help us assess what the optimal depth is for this decision tree.

```python
max_depth_list = list(range(1,9))
accuracy_scores = []

# Loop over the range of depth values, train and fit the model, and calculate accuracy
for depth in max_depth_list:
    regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate optimal depth (i.e. the depth that gives greatest accuracy)
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot max depth vs accuracy
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker="x", color="red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
```
The code above produces this plot of accuracy against max depth.
![Decision tree accuracy versus maximum depth](/img/posts/loyalty_decision_tree_max_depth_accuracy.png)

In the plot we can see that maximum accuracy is obtained by setting the maximum depth of the decision tree to 8. However, we can also see that accuracy does not improve much once max depth goes beyond 4. Using a max depth of 4 will give us a simpler model that will be easier to explain to our client and which should be more robust when making predictions on new data, so we decide to go with max depth of 4.

### Decision Tree Visualization
We can use the scikit-learn *plot_tree* functionality to create a visualization of our decision tree model with our chosen max depth of 4.
```python
regressor = DecisionTreeRegressor(random_state=42, max_depth=4)
regressor.fit(X_train, y_train)

# Create the visualization of the tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled=True,
                 rounded=True,
                 fontsize=16)
plt.show()
```
![Decision tree accuracy versus maximum depth](/img/posts/loyalty_decision_tree_visualization.png)

This is a really powerful visual that will really help us when we explain our modelling work to our client.

Interestingly the *distance_from_store* metric is used for the first split, suggesting (just as we saw in our linear regression model) that this variable plays an important role in determining points balances.

---

# Random Forest <a name="random-forest"></a>
The final model structure we will investigate is the random forest, which is where we create a group (an 'ensemble') of decision trees, with each tree being limited to using a randomly-assigned subset of the overall dataset and each decision node being limited to using a randomly-assigned subset of the input variables.

We'll once again use scikit-learn to implement the model, and here we'll break the code down into four sections:
- Data import
- Data preprocessing
- Model training
- Performance assessment

### Data Import
We again load the data that we saved in pickle format and shuffle it.
```python
# Import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# Import the pickle file we saved earlier and shuffle it
data_for_model = pickle.load(open("data/regression_modelling_data.p", "rb"))
data_for_model = shuffle(data_for_model, random_state=42)
```

### Data Preprocessing
##### Missing Values
Next, just as we did for our linear regression and decision tree models, we remove the small number of data points that have missing values.

```python
# Remove rows that have missing values
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```

##### Dealing With Outliers
Random forest models are generally pretty robust against the effects of outliers. However, we remove the two extreme outliers in our *points_balance* data (the balances of the CFO and the competition winner) as we we're confident that our model does not need to make predictions of extreme points balances, as our client is sure that the CFO and the competition winner were the only two people with such huge points balances.

```python
potential_outlier_vars = ["points_balance"]

for column in ["points_balance"]:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace=True)
```

##### Splitting Datasets for Modelling
We split the input variable out into its own DataFrame use an 80-20 split for the training and test data.
```python
# Split the data into input variables (X) and output variable (y)
X = data_for_model.drop(["points_balance"], axis=1)
y = data_for_model["points_balance"]

# Allocate 80% of data for model training leaving 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

##### Categorical Input Variables
Just like linear regression and decision tree models, random forest models cannot deal with data in categorical variables with string (here 'M' or 'F') categories, so we use one-hot encoding to turn our categorical variable into numeric form.

```python
# List of categorical variables that need encoding
categorical_vars = ["gender"]

# Create the encode, drop="first" to avoid multicollinearity
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Encode the categorical variable
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the names of the encoded features
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Replace the categorical variable with the encoded equivalent
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_vars, axis=1, inplace=True)
```

### Training the Model
Next we instantiate our random forest model and train it using our training set. We'll scikit-learn's default number of trees, which is 100.
```python
# Instantiate the random forest model
regressor = RandomForestRegressor(random_state=42)
# Fit the model to our training input and output variable data
regressor.fit(X_train, y_train)
```

### Model Performance
##### Make Predictions on the Test Set
We can now use our model (which we called *regressor*) to estimate the *points_balance* values for the test set.
```python
# Predict on the test set
y_pred = regressor.predict(X_test)
```

##### Calculate R-Squared, Cross-Validated R-Squared and Adjusted R-Squared
As for linear regression and decision tree, we calculate R-Squared using our predicted points balances for the test set (*y_pred*) and the accompanying known test set points balances (*y_test*). We then also calculate the cross-validated and adjusted R-Squared values.
```python
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Cross validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
print(cv_scores.mean())

# Calculate Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting R-Squared is a very impressive 0.957 and the cross-validated R-Squared is 0.925.

The adjusted R-Squared is 0.952 which is higher than the corresponding values for our linear regression and decision tree models, which were 0.769 and 0.898, respectively.

### Feature Importance and Permutation Importance
We are able to use our random forest to assess the importance of each input feature (input variable). There are two commonly-used ways we can do this:
- Feature Importance: For each of the input variables we sum up the impact of the split at all the nodes that use that input variable as their decision metric. In our case we're using the scikit-learn default metric for split quality in regression random forests, which is mean squared error (MSE). The greater the impact on MSE, the greater the importance of the input variable in the overall decision-making process of the model.
 - Permutation Importance: Permutation importance assesses the importance of each input variable by calculating the decrease in predictive performance when a variable's values are randomly shuffled (permutated). Since each tree in a random forest is limited to using a randomly-assigned subset of the training data we can use each tree's 'out-of-bag' data (the data not used to create the tree) to assess its performance.
-- We do this by first measuring the accuracy of the tree's performance when making predictions for the out-of-bag data points.
-- Then, for each input variable in turn, we permutate the variables values - that is, we randomly assign each value to a different data point in the out-of-bag data, in order to break the underlying relationship between the input variable and output variable. We then measure the tree's predictive accuracy for the permutated out-of-bag data and compare it to the accuracy we got with the un-permutated data.
-- We repeat the process for each tree in the forest to get an overall measure of how important the variable is in predicting the output variable. 

Permutation Importance is sometimes preferred over Feature Importance as the latter can, in some cases, exaggerate the importance of numeric variables in comparision to categorical variables. In most cases though the two approaches give similar results, so we'll use both and see what happens in our case.

Our code below runs the feature importance and permutation importance analyses and produces the two plots below.
```python
# Calculate feature importance
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X_train.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace=True)

# Plot feature importance
plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Calculate permutation importance 
result = permutation_importance(regressor, X_test, y_test, n_repeats=10, random_state=42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X_train.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis=1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by="permutation_importance", inplace=True)

# Plot permutation importance
plt.barh(permutation_importance_summary["input_variable"], permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()
```
![Random forest feature importance](/img/posts/loyalty_random_forest_feature_importance.png)
<br>
<br>
![Random forest permutation importance](/img/posts/loyalty_random_forest_permutation_importance.png)

Both analyses show that:
- *distance_from_store* is the most important variable in predicting *points_balance*, which is what we also saw with our linear regression and decision tree models.
- *credit_score* and *gender_M* are of little use in predicting *points_balance*. This confirms what we saw using RFECV, which suggested we should remove those two variables from our linear regression analysis.

---

# Modelling Summary <a name="modelling-summary"></a>
Understanding which variables drive points balances is important for our own understanding of the problem. The main requirement from our client is that we reliably estimate the loyalty points balances for the customers whose balances are missing. To that end we'll pick the model that gives us the highest accuracy, which is the random forest.

### Metric 1: Adjusted R-Squared (based on test set)
- Random Forest: 0.952
- Decision Tree: 0.898
- Linear Regression: 0.769

### Metric 2: R-Squared using K-Fold Cross-Validation (with k=4)
- Random Forest: 0.925
- Linear Regression: 0.847
- Decision Tree: 0.840

Our client did not ask for an analysis of the drives of customer loyalty points balances, but we will feed back to them our finding that *distance_from_store* is the most important variable in predicting a customer's points balance, which may be useful for the client to know going forward, especially if they are considering opening additional stores.

---

# Estimating the Missing Points Balances <a name="modelling-predictions"></a>
We're finally ready to estimate what the points balances should be for those customers whose points balances got wiped from our client's data centre. 
```python
# Import required packages
import pandas as pd
import pickle

# Import data we need to make predictions on
data_for_inference = pickle.load(open("data/regression_data_for_inference.p", "rb"))

# Import random forest model and OHE
regressor = pickle.load(open("data/random_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regression_ohe.p", "rb"))

# Drop rows with missing values
data_for_inference.dropna(how="any", inplace=True)

# Encode categorical variables
categorical_vars = ["gender"]
encoded_vars = one_hot_encoder.transform(data_for_inference[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
encoded_vars_df = pd.DataFrame(encoded_vars, columns=encoder_feature_names)
data_for_inference = pd.concat([data_for_inference.reset_index(drop=True), encoded_vars_df.reset_index(drop=True)], axis=1)
data_for_inference.drop(categorical_vars, axis=1, inplace=True)

# Estimate points balances!
points_balance_predictions = regressor.predict(data_for_inference)

# Calculate average points balance of our predictions
print(points_balance_predictions.mean())
```

The average of our points balances estimates turns out to be 7,455 points. Due to the impressive accuracy score on the test data we can be reasonably confident that our estimates will be acceptable to the supermarket's customers. Since the average points balance is less than half the 15,000 points that the panicked CEO was willing to give to each customer, our ML skills will have saved our client a lot of money, which is what we like to hear!

---

# Growth and Next Steps <a name="growth-next-steps"></a>
While our predictive accuracy was relatively high (95.2% based on our test set) we could look to further improve accuracy by tuning the random forest hyperparameters (such as maximum tree depth and the number of trees in the forest).

In terms of data, we could ask our client if their loyalty points have an expiration date and if they have any data that can give us insights into when each customer last visited a store. Our training data was based on customers who visited a store at least once in the last week, but the customers whose points balances we are estimating may no longer be regular customers and some of their points may have expired.
