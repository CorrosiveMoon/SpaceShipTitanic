
# Data Analysis Documentation

This documentation explains the steps taken to analyze the dataset using Python and various libraries such as Pandas, Matplotlib, and Seaborn.

## Loading the Dataset

The first step is to load the dataset from a CSV file using the `read_csv` function from Pandas. The dataset is stored in the variable `df`.

## Exploring the Dataset

To understand the dataset, we perform some initial exploratory data analysis.

- `df.head()`: Displays the first 5 rows of the dataset.
- `df.tail()`: Displays the last 5 rows of the dataset.
- `df.info()`: Provides information about the dataset, including column names, data types, and number of non-null values.
- `df.shape`: Prints the shape of the dataset (number of rows and columns).
- `df.describe()`: Generates summary statistics of the numerical columns in the dataset.
- `df.isnull().sum()`: Counts the number of null values in each column of the dataset.

## Handling Missing Values

As there are many null values in the dataset, we try to fill them by examining the relationships between the expenses and other variables. We calculate the correlation coefficients and find no strong correlation between the variables.

- `df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)`: Fills the null values in the 'RoomService' column with the mean of the column.
- Similar operations are performed for other columns: 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'.

## Dropping Duplicate Values

To remove duplicate rows from the dataset, we use the `drop_duplicates` function.

- `df.drop_duplicates(inplace=True)`: Drops duplicate rows from the dataset.

## Checking for Null Values

After filling the null values and dropping duplicates, we check again for any remaining null values in the dataset.

- `df.isnull().sum()`: Counts the number of null values in each column of the dataset.

## Frequency Analysis

We calculate the frequency of non-numerical variables to gain insights into their distributions.

- `planet_freq = df['HomePlanet'].value_counts()`: Calculates the frequency of each unique value in the 'HomePlanet' column.
- `cryo_freq = df['CryoSleep'].value_counts()`: Calculates the frequency of each unique value in the 'CryoSleep' column.
- `Cabin_freq = df['Cabin'].value_counts()`: Calculates the frequency of each unique value in the 'Cabin' column.
- `destination_freq = df['Destination'].value_counts()`: Calculates the frequency of each unique value in the 'Destination' column.
- `vip_freq = df['VIP'].value_counts()`: Calculates the frequency of each unique value in the 'VIP' column.

## Outlier Detection

We identify outliers and inconsistencies in the dataset using box plots for each numerical column.

- For each numerical column, a box plot is generated using `df.boxplot([col])`.

## Data Visualization

We create various visualizations to gain insights into the data.

- `df['Transported'].value_counts().plot(kind='bar')`: Bar chart showing the frequency of transported passengers.
- `df['Age'].hist(bins=10)`: Histogram displaying the distribution of passenger ages.
- `df['CryoSleep'].value_counts().plot(kind='bar')`: Bar chart showing the frequency of cryosleep.
- `df['HomePlanet'].value_counts().plot(kind='bar')`: Bar chart showing the frequency of home planets.
- `df['Destination'].value_counts().plot(kind='bar')`: Bar chart showing the frequency of destinations.
- `df['VIP'].value_counts().plot(kind='bar')`: Bar chart showing the frequency of VIP status.

## Correlation Analysis

We perform correlation analysis between variables to identify any relationships.

- `sns.jointplot(x='Age', y='Transported', data=df, kind='reg')`: Scatter plot with a regression line showing the relationship between age and transportation.
- `sns.scatterplot(x='Age', y='Transported', data=df)`: Scatter plot showing the relationship between age and transportation.
- `correlation(df['Age'], df['Transported'])`: Calculates the correlation coefficient between age and transportation.
- `correlation(df['IsVIP'], df['Transported'])`: Calculates the correlation coefficient between VIP status and transportation.

# Calculate the percentage of passengers transported for each unique value in the HomePlanet column
HomePlanet_Transported = df.groupby('HomePlanet')['Transported'].sum() / df.groupby('HomePlanet')['Transported'].count() * 100

# Calculate the percentage of passengers transported for each unique value in the CryoSleep column
CryoSleep_Transported = df.groupby('CryoSleep')['Transported'].sum() / df.groupby('CryoSleep')['Transported'].count() * 100

# Calculate the percentage of passengers transported for each unique value in the Destination column
Destination_Transported = df.groupby('Destination')['Transported'].sum() / df.groupby('Destination')['Transported'].count() * 100

# Calculate the percentage of passengers transported for each unique value in the VIP column
VIP_Transported = df.groupby('VIP')['Transported'].sum() / df.groupby('VIP')['Transported'].count() * 100

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 15))

# Plot a bar chart of the percentage of passengers transported for each unique value in the HomePlanet column
axs[0, 0].bar(HomePlanet_Transported.index, HomePlanet_Transported.values)
axs[0, 0].set_xticklabels(HomePlanet_Transported.index, rotation=90)
axs[0, 0].set_title('Transported Percentage by Home Planet')

# Plot a bar chart of the percentage of passengers transported for each unique value in the CryoSleep column
axs[0, 1].bar(CryoSleep_Transported.index, CryoSleep_Transported.values)
axs[0, 1].set_xticklabels(CryoSleep_Transported.index, rotation=90)
axs[0, 1].set_title('Transported Percentage by Cryo Sleep')

# Plot a bar chart of the percentage of passengers transported for each unique value in the Destination column
axs[1, 0].bar(Destination_Transported.index, Destination_Transported.values)
axs[1, 0].set_xticklabels(Destination_Transported.index, rotation=90)
axs[1, 0].set_title('Transported Percentage by Destination')

# Plot a bar chart of the percentage of passengers transported for each unique value in the VIP column
axs[1, 1].bar(VIP_Transported.index, VIP_Transported.values)
axs[1, 1].set_xticklabels(VIP_Transported.index, rotation=90)
axs[1, 1].set_title('Transported Percentage by VIP')

# Adjust the spacing between the subplots and display the plot
plt.tight_layout()
plt.show()

# About 64% of the Passengers from Europa were Transported
# About 78% of the Passengers in CryoSleep were transported
# The proportion of Passengers debarking to 55 Cancri e transported to another dimension is greater compared to those debarking to PSO J318.5–22 and TRAPPIST-1e
# About 38% of the Passengers that paid for special VIP services were transported

CabinDeck_Transported = df.groupby('CabinDeck').aggregate({'Transported': 'sum',
                                                           'PassengerId': 'size'
                                                          }).reset_index()

CabinDeck_Transported['TransportedPercentage'] = CabinDeck_Transported['Transported'] / CabinDeck_Transported['PassengerId']

CabinSide_Transported = df.groupby('CabinSide').aggregate({'Transported': 'sum',
                                                           'PassengerId': 'size'
                                                          }).reset_index()

CabinSide_Transported['TransportedPercentage'] = CabinSide_Transported['Transported'] / CabinSide_Transported['PassengerId']

# Visualize Cabin features vs target variable
plt.figure(figsize=(14, 15)) 
plt.subplot(221)
sns.barplot(x="CabinDeck", y="TransportedPercentage", data=CabinDeck_Transported, order=CabinDeck_Transported.sort_values('TransportedPercentage', ascending=False).CabinDeck)
plt.subplot(222)
sns.barplot(x="CabinSide", y="TransportedPercentage", data=CabinSide_Transported, order=CabinSide_Transported.sort_values('TransportedPercentage', ascending=False).CabinSide)

# Create a new column called Alone that indicates whether each passenger is traveling alone or not
df["PassengerGroup"] = df["PassengerId"].str.split('_', expand=True)[0]
No_People_In_PassengerGroup = df.groupby('PassengerGroup').aggregate({'PassengerId': 'size'}).reset_index()
No_People_In_PassengerGroup = No_People_In_PassengerGroup.rename(columns={"PassengerId": "NoInPassengerGroup"})
No_People_In_PassengerGroup["IsAlone"] = No_People_In_PassengerGroup["NoInPassengerGroup"].apply(lambda x: "Not Alone" if x > 1 else "Alone")
df = pd.merge(df, No_People_In_PassengerGroup[["PassengerGroup", "IsAlone"]], on="PassengerGroup")

IsAlone_Transported = df.groupby('IsAlone').aggregate({'Transported': 'sum',
                                                       'PassengerId': 'size'
                                                      }).reset_index()

# create dataframe IsAlone_Transported that contains percentage of passengers transported Alone or Not Alone
IsAlone_Transported['TransportedPercentage'] = IsAlone_Transported['Transported'] / IsAlone_Transported['PassengerId']

# Visualize IsAlone vs transported
sns.barplot(x="IsAlone", y="TransportedPercentage", data=IsAlone_Transported, order=IsAlone_Transported.sort_values('TransportedPercentage', ascending=False).IsAlone)

df = df.drop(['CabinDeck', 'CabinNo.', 'CabinSide', 'PassengerGroup'], axis=1)

# Encode categorical variables
le = LabelEncoder()
df['HomePlanet'] = le.fit_transform(df['HomePlanet'])
df['Cabin'] = le.fit_transform(df['Cabin'])
df['IsAlone'] = le.fit_transform(df['IsAlone'])
df['Destination'] = le.fit_transform(df['Destination'])

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.3, random_state=22)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Create models and evaluate their performance
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for i in range(len(models)):
    models[i].fit(X_train, Y_train)
  
    print(f'{models[i]}:')
  
    train_preds = models[i].predict_proba(X_train)[:, 1]
    print('Training AUC-ROC Score:', ras(Y_train, train_preds))
  
    val_preds = models[i].predict_proba(X_val)[:, 1]
    print('Validation AUC-ROC Score:', ras(Y_val, val_preds))
    print()

## Conclusion 

The code provided performs data loading, exploration, handling of missing values, duplicate removal, frequency analysis, outlier detection, data visualization, and correlation analysis. These steps help in understanding the dataset and identifying patterns and relationships between variables.


