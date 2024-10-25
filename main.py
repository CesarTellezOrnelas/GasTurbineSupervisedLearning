import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 150)

# fetch dataset
gas_turbine_co_and_nox_emission_data_set = fetch_ucirepo(id=551)

# save data as feature and target dataframes
feature_df = gas_turbine_co_and_nox_emission_data_set.data.features
target_df = gas_turbine_co_and_nox_emission_data_set.data.targets

# metadata
print(gas_turbine_co_and_nox_emission_data_set.metadata)
print("---------------------------------------------------------------------------------------------------------------")
# features table structure (Turbine Energy Yield, TEY, is the target variable)
print(feature_df)
print("---------------------------------------------------------------------------------------------------------------")

# TEY values histogram for all data
plt.hist(feature_df['TEY'], bins=30, color='blue', edgecolor='black')
plt.ylabel('Frequency')
plt.xlabel('TEY Value (MWH)')
plt.title('Histogram of TEY Values for All Data')
plt.show()

# Need to split dataset into first 3 years (2011-2013 for training) and the last 2 years (2014 to 2015 for testing)
training_set = feature_df[feature_df['year'].between(2011, 2013)] # training
testing_set = feature_df[feature_df['year'].between(2014, 2015)] # testing

# Prepare the data by removing target and unnecessary year columns. Define target datasets
feature_training_set = training_set.drop(columns=['TEY', 'year'])
feature_testing_set = testing_set.drop(columns=['TEY', 'year'])
target_training_set = training_set['TEY']
target_testing_set = testing_set['TEY']

# Choose a regression model
model = LinearRegression()

# Train the model
model.fit(feature_training_set, target_training_set)

# Evaluate the model
target_prediction = model.predict(feature_testing_set)
# Overall error magnitude, smaller is better.
mse = mean_squared_error(target_testing_set, target_prediction)
# root mean squared error, for context
rmse = math.sqrt(mse)
# How well the model fits the data, closer to 1 is better
r2 = r2_score(target_testing_set, target_prediction)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score: ", r2)

# Calculate residuals and plot in a residplot
residuals = target_testing_set - target_prediction
plt.figure(figsize=(10, 6))
sns.residplot(x=target_testing_set, y=residuals, lowess=True, scatter_kws={'alpha': 0.5, 'color':'black'}, line_kws={'color': 'red', 'lw': 2})
plt.title('Residual Plot with Lowess Smoothing')
plt.xlabel('Actual TEY')
plt.ylabel('Residuals')
plt.axhline(y=0, color='blue', linestyle='--')  # Reference line at 0
plt.show()

# Access the coefficients, or weights
coefficients = model.coef_

# Print the coefficients
print("Coefficients: ", coefficients)