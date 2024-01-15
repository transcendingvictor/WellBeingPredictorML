# %%
# load and show the dataset 
import pandas as pd
pd.set_option('display.max_columns', None) # see all columns in scrollable way

df = pd.read_csv("jkl_indresp.tab",  sep='\t', encoding="ISO-8859-1")
df_original = df.copy()
print(df.shape) #(31847, 1853)
df.head(10)
# column_list = df_UK.columns.tolist()
# print(column_list)
# %%
"""
The actual meaning of each column can be found in the corresponding [...] pdf.
As it is the result of a survey, "missing values" have different categories like
"Unapplicable", "Don't know", "...".  All of these have negative values associated to
them, while non-negative values are left for actual responses.
"""
# %%
# defines a function that drops columns with high percentage of missing values

def remove_missing_columns(df, threshold_percentage):
    """
    Removes columns from a DataFrame where the percentage of negative values
    exceeds the given threshold.

    :param df: Pandas DataFrame to process
    :param threshold_percentage: Threshold for percentage of negative values (0-100)
    :return: DataFrame with columns removed
    """
    # Calculate the percentage of negative values in each column
    percentage_negative = (df < 0).mean() * 100

    # Find columns where the percentage of negative values is greater than the threshold
    columns_to_drop = percentage_negative[percentage_negative > threshold_percentage].index

    # Drop these columns
    df_filtered = df.drop(columns=columns_to_drop)

    return df_filtered
# %%
# I iterated through possible thresholds and saw that the difference between 70 and 25 wasnt
# very big, which indicates that many of the coluns had a lot of missing values, but the rest
# didnt. So I put a high threshold.

for threshold in [99, 95, 90, 75, 60, 50, 40, 30, 25]:
    df_UK_filtered = remove_missing_columns(df, threshold)
    print(f"\n If we removed columns with more than {threshold}% of missing values, we'd keep {df_UK_filtered.shape[1]: .0f} columns \n which is {df_UK_filtered.shape[1]/1853*100: .2f}% of them")
# %%
df = remove_missing_columns(df, 25)
print(df.shape)
df.head()
# %%
#DROPPING COLUMNS (TO-DO make lists, then dropp them ALTOGETHER)

# remove all columns before sex
sex_index = df.columns.get_loc("jkl_sex")
df = df.iloc[:, sex_index:]
print(df.shape)
df.sample(10)

#now all columns with "dat"
date_columns = [col for col in df.columns if "dat" in col]
df = df.drop(columns=date_columns,
                                       errors="ignore")

#now all columns with chk
check_columns = [col for col in df.columns if "chk" in col]
df = df.drop(columns=check_columns,
                                       errors="ignore")

# now all columns whose definition is not clear from info or present or past interview details
miscelldrop_columns = ["jkl_indmode", "jkl_ff_ivlolw", "jkl_hrpid", 'jkl_doby_if', 'jkl_j2pay_if',
                       'jkl_seiss_amount_if', 'jkl_paygu_if', 'jkl_paynu_if', 'jkl_seearngrs_if', 
                       'jkl_fiyrinvinc_if', 'jkl_fibenothr_if', 'jkl_fimnlabgrs_if', 'jkl_fimngrs_if',
                       'jkl_hgpart', 'jkl_hgbiom', 'jkl_hgbiof', 'jkl_hgadoptm', 'jkl_hgadoptf',
                       'jkl_deviceused', 'jkl_ideviceused', 'jkl_birthy', 'jkl_pbirthy']
# here I also dropped data related to year of birth (as its equivalent to the age after scaling)

df = df.drop(columns=miscelldrop_columns,
                                       errors="ignore")

# %%
#no constant columns
constant_columns = [col for col in df.columns if
                     df[col].nunique()==1]
print(len(constant_columns))
# %%
#plot the number of columns per number of unique values they have

import matplotlib.pyplot as plt
import numpy as np
#superimportant (a lot of missing values)
df = df.applymap(lambda x: np.nan if x < 0 else x)

# Calculate the number of unique values for each column

unique_counts = df.apply(lambda col: col.nunique())

# Plot a histogram
plt.figure(figsize=(10, 6))
plt.hist(unique_counts, bins=range(31), edgecolor='black', align='left')
plt.title('Histogram of Unique Value Counts per Column')
plt.xlabel('Number of Unique Values')
plt.ylabel('Number of Columns')
plt.xticks(range(31))
plt.grid(axis='y', alpha=0.75)
plt.show()
**+# %%
def count_nonnegative_unique_values(col):
    return col[col >= 0].nunique()

# Apply this function to each column in the DataFrame
unique_counts_nonnegative = df.apply(
    count_nonnegative_unique_values)

print(unique_counts_nonnegative.shape)
unique_counts_nonnegative.head()
# %%
somewhat_constant_columns = [col for col in
                             unique_counts_nonnegative.index if
                             unique_counts_nonnegative[col]==1]
# 8 columns of the "person1,2,3...16" for some reason
# %%
#dropping columns to prevent data leakage (mental health, satisfaction)

# List of columns to drop
columns_to_drop = []

# Adding jkl_sclfsatX columns
columns_to_drop.extend([f'jkl_sclfsat{num}' for num in [1, 2, 7]])

# Adding jkl_hcondcodeX columns
columns_to_drop.extend([f'jkl_hcondcode{num}' 
                        for num in range(37, 44)])

# Assuming jkl_scghqX columns go from 'a' to 'l'
letters = [chr(i) for i in range(ord('a'), ord('l') + 1)]
columns_to_drop.extend([f'jkl_scghq{letter}' for letter in letters])

# Adding jkl_scghq1_dvX columns
columns_to_drop.extend([f'jkl_scghq{num}_dv' for num in [1, 2]])

columns_to_drop.extend(["jkl_scghq2_dv", "jkl_scghq1_dv"])

df = df.drop(columns=columns_to_drop,
                                       errors="ignore")

# %%

# %%
#checking the values for the life satisfaction column
unique_values = df['jkl_sclfsato'].unique()
print("Unique values in 'jkl_sclfsato':", sorted(unique_values))


unique_counts = df.apply(lambda col: col.nunique())

# Plot a histogram
plt.figure(figsize=(10, 6))
df['jkl_sclfsato'].value_counts().sort_index().plot(kind='bar')
plt.title('Frequencies for Life Satisfaction')
plt.xlabel('Life satisfaction Score')
plt.ylabel('Number of Rows')
plt.xticks(rotation=0)
plt.show()

life_satisfaction_counts = df['jkl_sclfsato'].value_counts().to_dict()
sorted_dict = dict(sorted(life_satisfaction_counts.items(), key=lambda item: item[0], reverse=True))
print(sorted_dict)

#the frequencies
total = sum(sorted_dict.values())
percentages = {k: round(v/total*100, 2) for k,v in sorted_dict.items()}
print(percentages)
# %%
#removing the rows with missing life satisfaction column
import numpy as np
missing_value_codes = [-9, -8, -7, -2, -1]
df['jkl_sclfsato'].replace(missing_value_codes, np.nan, inplace=True)

df.dropna(subset=['jkl_sclfsato'], inplace=True)

#now there should just be values from 1 to 7
unique_values = df['jkl_sclfsato'].unique()
print("Unique values in 'jkl_sclfsato':", sorted(unique_values))
print(df.shape) #30919 rows

# %%
#LIST OF COLUMNS (search method to find the categorical variables)
# Assuming df_UK_nomissing is your DataFrame
unique_two_value_columns = []
other_columns = []

df = df.applymap(lambda x: np.nan if x < 0 else x)


# Iterate through each column to check the number of unique non-missing values
for column in df.columns:
    non_missing_values = df[column].dropna().unique()
    if len(non_missing_values) == 2:
        unique_two_value_columns.append(column)
    else:
        other_columns.append(column)

# Now, unique_two_value_columns list contains the names of columns with exactly two unique non-missing values
# other_columns list contains the rest of the columns

# To view the DataFrame without the two-value columns:
df_other_columns = df[other_columns]

#unique_two_values should be left as 0 and 1 values.
#missing values go to the most common value of the two
# Display the DataFrame with the selected columns
#say that all these variables are categorical because...
# ... "numerical variables in the survey had more options or sth"

print(df_other_columns.shape)


# %%
categorical_columns = ["jkl_jbstat", "jkl_ff_jbstat", # economic activity
                       "jkl_vote7", "jkl_vote3", "jkl_euparl", "jkl_euref", #political party or VOTED/DIDN'T_VOTE/COULDN'T_VOTE
                       "jkl_marstat_dv", "jkl_currmstat", #marital status
                       "jkl_qfhigh_dv", "jkl_hiqual_dv", #highest education qualification
                       "jkl_ctrel", #most stable contact relation  (son, father, friend...)
                       ]
# variables of clasess like jkl_hl2gp are numerical although not exact numbers
# %%
encoded_columns = categorical_columns + unique_two_value_columns
#encoding unique_two_value_columns is one of the ways to make it 0 and 1
#first, give missing values their own column (1 if missing, 0 otherwise)
for column in categorical_columns:
    df[column + '_missing'] = np.where(df[column].isnull(), 1, 0)

# then, create the dummy variables for each column ()
df = pd.get_dummies(df, columns=encoded_columns)


# %%
#some columns will not be standarized by themselves, but rather their logs
log_columns = ['jkl_fimngrs_dv', "jkl_fimnlabgrs_dv", "jkl_fimnlabnet_dv", "jkl_fiyrinvinc_dv",
                "jkl_fibenothr_dv", "jkl_fimnmisc_dv", "jkl_fimnprben_dv", "jkl_fimninvnet_dv",
                  "jkl_fimnpen_dv", "jkl_fimnsben_dv", "jkl_fimnnet_dv"]  #all income related
                  # "jkl_nkids_dv" number of kids do i include it?

df[log_columns] = np.log(df[log_columns]+1)
#we add the relatively small value +1 to avoid log(0) which is undefined

# as we'll do min-max scaling, this is more appropiate for the high variables.
# %%
#scaling the numerical columns
from sklearn.preprocessing import MinMaxScaler

numerical_columns = df.columns.difference(categorical_columns)

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

#missing values assigned the observed mean
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

df.head()


# %%
#checking for duplicated columns
transposed_df = df.T  #transposed to use the duplicate() method

# Find duplicated rows (which were originally columns)
duplicated_columns = transposed_df[transposed_df.duplicated()].index.tolist()

print("Duplicate columns:", duplicated_columns)
df = df.drop(duplicated_columns, errors="ignore")
# %%
#split the dataset
from sklearn.model_selection import train_test_split


y = df["jkl_sclfsato"]
X = df.drop("jkl_sclfsato", axis=1)

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



# %% Linear regression through OLS
#R^2 square of OLS (0.42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict the values
predictions = model.predict(X_test)

# Calculate R-squared score (R2: 0.416)
r_squared = r2_score(y_test, predictions)
print(f'R-squared: {r_squared}')

#Calculate the RMSE (RMSE: 0.179)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
print(f"RMSE: {np.sqrt(mse)}")
# %%
#linear regression with gradient descent
from sklearn.linear_model import SGDRegressor

model_GD = SGDRegressor(loss="squared_error", learning_rate="constant", eta0=0.0015, max_iter=1000, tol=1e-3)
model_GD.fit(X_train, y_train)

predictions = model_GD.predict(X_test)
# Calculate R-squared score (R2: 0.423)
r_squared = r2_score(y_test, predictions)
print(f'R-squared: {r_squared}')

#Calculate the RMSE (RMSE: 0.175)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
print(f"RMSE: {np.sqrt(mse)}")
# %%
#Accuracy of the trivial regressor

mean_satisfaction = np.mean(y_train)
predictions = np.full(shape=y_test.shape, fill_value=mean_satisfaction)

# Calculate R-squared score
r_squared = r2_score(y_test, predictions)
print(f'R-squared: {r_squared}')

#Calculate the MSE (RMSE: 0.23)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
print(f"RMSE: {np.sqrt(mse)}")
# %%
#decission tree classifier
from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor(random_state=30)
dt_regressor.fit(X_train, y_train)
dt_predictions = dt_regressor.predict(X_test)

# Calculate Mean Squared Error (RMSE=0.255) 
mse = mean_squared_error(y_test, dt_predictions)
print(f'MSE: {mse}')


# Calculate R-squared 
r2 = r2_score(y_test, dt_predictions)
print(f'R-squared: {r2}')

# %+-
#support vector machine regressor (takes too long, never got results...)
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline


# svr_regressor = make_pipeline(StandardScaler(), SVR())

# svr_regressor.fit(X_train, y_train)
# svr_predictions = svr_regressor.predict(X_test)
# # Calculate R-squared score
# r_squared = r2_score(y_test, svr_predictions)
# print(f'R-squared: {r_squared}')

# #Calculate the MSE (RMSE: 0.23)
# mse = mean_squared_error(y_test, svr_predictions)
# print(f"MSE: {mse}")
# print(f"RMSE: {np.sqrt(mse)}")
# %% 
from sklearn.neighbors import KNeighborsRegressor

knn_regressor = KNeighborsRegressor(n_neighbors=25) #3, 5, 10, 25
knn_regressor.fit(X_train, y_train)
knn_predictions = knn_regressor.predict(X_test)

# Calculate Mean Squared Error (RMSE=0.255, 0.255, 0.255, 0.255)
mse_knn = mean_squared_error(y_test, knn_predictions)
print(f'MSE (KNN): {mse_knn}')
print(f"RMSE: {np.sqrt(mse)}")

# Calculate R-squared (R²=0.8, 0.14, 0.17 )
r2_knn = r2_score(y_test, knn_predictions)
print(f'R-squared (KNN): {r2_knn}')

# %%
# Decission Tree with 100 tress (a bit less than 5 mins)
from sklearn.ensemble import RandomForestRegressor

#play with max_depth and other params
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_rf = mean_squared_error(y_test, rf_predictions)
print(f'MSE (Random Forest): {mse_rf}')
print(f"RMSE: {np.sqrt(mse)}")

# Calculate R-squared (R²=0.403)
r2_rf = r2_score(y_test, rf_predictions)
print(f'R-squared (Random Forest): {r2_rf}')

# %%
