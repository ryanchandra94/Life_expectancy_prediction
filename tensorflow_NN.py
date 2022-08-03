import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# Import data from csv file
df = pd.read_csv("Life_Expectancy_Data.csv", index_col=False)
####Drop unnecessary value & pick prediction target
df_clean = df.drop(["Country","Continent","Year","Status","Life_expectancy"], axis=1)
df_target = df["Life_expectancy"]
target_mean = df_target.mean()
#Clean the data ("Unknown data to Nan")
df_clean["Diphtheria"] = pd.to_numeric(df_clean['Diphtheria'])
df_clean["Population"] = pd.to_numeric(df_clean['Population'])
# Get the mean value for every column that have NaN value
hepatitis_mean = df_clean["Hepatitis_B"].mean()
polio_mean = df_clean["Polio"].mean()
total_expenditure_mean = df_clean["Total_expenditure"].mean()
income_composition_mean = df_clean['Income_composition_of_resources'].mean()
schooling_mean = df_clean["Schooling"].mean()
diphtheria_mean = df_clean["Diphtheria"].mean()
population_mean = df_clean['Population'].mean()
# Replace NaN value with the mean from its column
df_clean["Hepatitis_B"].fillna(value=hepatitis_mean, inplace=True)
df_clean["Polio"].fillna(value=polio_mean, inplace=True)
df_clean["Total_expenditure"].fillna(value=total_expenditure_mean, inplace=True)
df_clean["Income_composition_of_resources"].fillna(value=income_composition_mean, inplace=True)
df_clean["Schooling"].fillna(value=schooling_mean, inplace=True)
df_clean["Population"].fillna(value=population_mean, inplace=True)
df_clean["Diphtheria"].fillna(value=diphtheria_mean, inplace=True)

df['New Column'] = np.where(df['Life_expectancy'] >= target_mean, 1, 0)

#df_immunization = df_clean[["Polio", "Diphtheria", "Measles", "Hepatitis_B"]]
#df_mortality = df_clean[["Adult_Mortality", "infant_deaths", "under_five_deaths"]]
#df_economic = df_clean[["percentage_expenditure", "Total_expenditure", "Income_composition_of_resources", "GDP"]]
#df_social = df_clean[["Population", "Schooling", "Alcohol"]]
#df_others = df_clean[["BMI", "HIV/AIDS", "thinness  1-19 years", "thinness 5-9 years"]]


x = df_clean.values
y = pd.get_dummies(df["New Column"]).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=4)

model = Sequential()
model.add(Dense(20, input_dim=18, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1500)

accuracy = model.evaluate(x, y)

print (accuracy)