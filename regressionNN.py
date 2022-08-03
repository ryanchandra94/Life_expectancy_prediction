import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy
from iteration_utilities import deepflatten

# Import data from csv file
from Life_Expectancy import hidden_layer_error

df = pd.read_csv("Life_Expectancy_Data.csv", index_col=False)
df_clean = df.drop(["Country", "Continent", "Year", "Status", "Life_expectancy"], axis=1)
df_target = df["Life_expectancy"]
target_mean = df_target.mean()
# Clean the data ("Unknown data to Nan")
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

x = df_clean.values
y = df_target.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=4)


def relu(x):
    number = np.maximum(0, x)
    return number

def derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def mean_squarred_error(prediction, labels):
    N = labels.size
    mse = ((prediction - labels) ** 2).sum() / (2 * N)
    return mse


learning_rate = 0.1
N = y_train.size  # number of example x input nodes
epochs = 1

n_input = 18
n_hidden = 10
n_output = 1

np.random.seed(10)

weight1 = np.random.normal(scale=0.5, size=(n_input, n_hidden))
weight2 = np.random.normal(scale=0.5, size=(n_hidden, n_output))

monitoring = {"mean_squared_error": [], "result": []}

for epoch in range(epochs):
    # feedforward
    hidden_layer_inputs = np.dot(x_train, weight1)
    hidden_layer_outputs = relu(hidden_layer_inputs)

    output_layer_inputs = np.dot(hidden_layer_outputs, weight2)
    output_layer_outputs = relu(output_layer_inputs)
    list_result = output_layer_outputs.tolist()
    new_result = list(deepflatten(list_result))
    print(y_train)
    print("Result: \n")
    print(new_result)

    # monitor training process
    mse = mean_squarred_error(new_result, y_train)

    monitoring["mean_squared_error"].append(mse)
    monitoring["result"].append(new_result)

    # backpropagation
    output_layer_error = new_result - y_train
    output_layer_delta = derivative_relu(output_layer_error)

    # hidden_layer_error = np.dot(output_layer_delta, weights_2.T)
    # hidden_layer_delta = derivative_relu(hidden_layer_error)

monitoring_df = pd.DataFrame(monitoring)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error")

print(monitoring_df['mean_squared_error'])
# plt.show()
