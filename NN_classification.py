
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
# Import data from csv file
df = pd.read_csv("Life_Expectancy_Data.csv", index_col=False)
####Drop unnecessary value & pick prediction target
df_clean = df.drop(["Country","Continent","Year","Status","Life_expectancy","Hepatitis_B", "Population", "Measles"], axis=1)
df_target = df["Life_expectancy"]
target_mean = df_target.mean()
#Clean the data ("Unknown data to Nan")
df_clean["Diphtheria"] = pd.to_numeric(df_clean['Diphtheria'])
#df_clean["Population"] = pd.to_numeric(df_clean['Population'])
# Get the mean value for every column that have NaN value
#hepatitis_mean = df_clean["Hepatitis_B"].mean()
polio_mean = df_clean["Polio"].mean()
total_expenditure_mean = df_clean["Total_expenditure"].mean()
income_composition_mean = df_clean['Income_composition_of_resources'].mean()
schooling_mean = df_clean["Schooling"].mean()
diphtheria_mean = df_clean["Diphtheria"].mean()
#population_mean = df_clean['Population'].mean()
# Replace NaN value with the mean from its column
#df_clean["Hepatitis_B"].fillna(value=hepatitis_mean, inplace=True)
df_clean["Polio"].fillna(value=polio_mean, inplace=True)
df_clean["Total_expenditure"].fillna(value=total_expenditure_mean, inplace=True)
df_clean["Income_composition_of_resources"].fillna(value=income_composition_mean, inplace=True)
df_clean["Schooling"].fillna(value=schooling_mean, inplace=True)
#df_clean["Population"].fillna(value=population_mean, inplace=True)
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

def sigmoid(x):
    return 1/(1+np.exp(scipy.special.expit(-x)))

def mean_squarred_error(prediction, labels):
    N = labels.size
    mse = ((prediction - labels)**2).sum() / (2*N)
    return mse

def accuracy(predictions, labels):
    predictions_correct = predictions.argmax(axis=1) == labels.argmax(axis=1)
    accuracy = predictions_correct.mean()
    return accuracy



learning_rate= 0.1
N = y_train.size #number of example x input nodes
epochs = 15000

n_input = 15
n_hidden1 = 20
n_hidden2 = 20
n_output = 2

np.random.seed(10)

weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden1))
weights_2 = np.random.normal(scale=0.5, size=(n_hidden1, n_hidden2))
weights_3 = np.random.normal(scale=0.5, size=(n_hidden2, n_output))

monitoring = {"mean_squared_error": [], "accuracy": []}


for epoch in range(epochs):
    #feedforward
    hidden_layer_inputs1 = np.dot(x_train,weights_1)
    hidden_layer_outputs1 = sigmoid(hidden_layer_inputs1)

    hidden_layer_inputs2 = np.dot(hidden_layer_outputs1, weights_2)
    hidden_layer_outputs2 = sigmoid(hidden_layer_inputs2)

    output_layer_inputs = np.dot(hidden_layer_outputs2, weights_3)
    output_layer_outputs = sigmoid(output_layer_inputs)



    #monitor training process
    mse = mean_squarred_error(output_layer_outputs, y_train)
    acc = accuracy(output_layer_outputs, y_train)

    monitoring["mean_squared_error"].append(mse)
    monitoring["accuracy"].append(acc)



    #backpropagation
    output_layer_error = output_layer_outputs - y_train
    output_layer_delta = output_layer_error * output_layer_outputs * (1-output_layer_outputs)

    hidden_layer_error2 = np.dot(output_layer_delta, weights_3.T)
    hidden_layer_delta2 = hidden_layer_error2 * hidden_layer_outputs2 * (1-hidden_layer_outputs2)

    hidden_layer_error1 = np.dot(hidden_layer_delta2, weights_2.T)
    hidden_layer_delta1 = hidden_layer_error1 * hidden_layer_outputs1 * (1-hidden_layer_outputs1)

    #weights update

    weights_3_update = np.dot(hidden_layer_outputs2.T, output_layer_delta) / N
    weights_2_update = np.dot(hidden_layer_outputs1.T, hidden_layer_delta2) / N
    weights_1_update = np.dot(x_train.T, hidden_layer_delta1) / N

    weights_3 = weights_3 - learning_rate * weights_3_update
    weights_2 = weights_2 - learning_rate * weights_2_update
    weights_1 = weights_1 - learning_rate * weights_1_update

monitoring_df = pd.DataFrame(monitoring)

fig, axes = plt.subplots(1, 2, figsize=(15,5))

monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error")
monitoring_df.accuracy.plot(ax=axes[1], title="Accuracy")
#print(monitoring_df.loc[monitoring_df["accuracy"] == 0.6614754098360656])
print("Neural network (2 hidden layers classification):")
print("Mean Squared Error:" + str(monitoring_df["mean_squared_error"].min()))
print("accuracy:" + str(monitoring_df["accuracy"].max()))
plt.show()

