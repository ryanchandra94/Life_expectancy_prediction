import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy
from iteration_utilities import deepflatten
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Life_Expectancy_Data.csv", index_col=False)
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


x = df_clean.values
y = df_target.values

target = [[i] for i in y]



x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=20, random_state=4)
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

def relu(x):
    number = np.maximum(0,x)
    return number


def derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x



def mean_squarred_error(prediction, labels):
    N = len(labels)
    mse = ((prediction - labels)**2).sum() / (2*N)
    return mse

learning_rate= 0.000000001
N = len(y_train) #number of example x input nodes
epochs = 3000

n_input = 18
n_hidden = 10
n_hidden2 = 10
n_output = 1

np.random.seed(10)

weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden))
weights_2 = np.random.normal(scale=0.5, size=(n_hidden, n_hidden2))
weights_3 = np.random.normal(scale=0.5, size=(n_hidden2, n_output))


#print(x_train)
monitoring = {"epoch": [], "mean_squared_error": [], "result": [],
              "mean_absolute_error": [], "root_mean_squared_error": []}


for epoch in range(epochs):


    #feedforward

    hidden_layer_inputs1 = np.dot(X_train,weights_1)
    #print("hidden_layer_inputs:")
    #print(hidden_layer_inputs)
    hidden_layer_outputs1 = relu(hidden_layer_inputs1)
    #print("hidden_layer_ouputs:")
    #print(hidden_layer_outputs)

    hidden_layer_inputs2 = np.dot(hidden_layer_outputs1, weights_2)
    #print("output_layer_inputs:")
    #print(output_layer_inputs)

    hidden_layer_outputs2 = relu(hidden_layer_inputs2)

    output_layer_inputs = np.dot(hidden_layer_outputs2, weights_3)

    output_layer_outputs = relu(output_layer_inputs)
    #print("output_layer_outputs:")
    #print(output_layer_outputs)
    #print("weights1:")
    #print(weights_1)

    #monitor training process

    mse = mean_squarred_error(output_layer_outputs, y_train)
    mae = mean_absolute_error(output_layer_outputs, y_train)
    rmse = np.sqrt(mse)

    monitoring["epoch"].append(epoch)
    monitoring["mean_squared_error"].append(mse)
    monitoring["result"].append(output_layer_outputs)
    monitoring["mean_absolute_error"].append(mae)
    monitoring["root_mean_squared_error"].append(rmse)

    # backpropagation
    output_layer_error = output_layer_outputs - y_train
    #print("output_layer_error:")
    #print(output_layer_error)
    output_layer_delta2 = output_layer_error * derivative_relu(output_layer_outputs)
    #print("output_layer_delta:")
    #print(output_layer_error)
    hidden_layer_error2 = np.dot(output_layer_delta2, weights_3.T)
    hidden_layer_delta2 = hidden_layer_error2 * derivative_relu(hidden_layer_outputs2)


    hidden_layer_error1 = np.dot(hidden_layer_delta2, weights_2.T)
    hidden_layer_delta1 = hidden_layer_error1 * derivative_relu(hidden_layer_outputs1)

    # weights update

    weights_3_update = np.dot(hidden_layer_outputs2.T, output_layer_delta2) / N
    weights_2_update = np.dot(hidden_layer_outputs1.T, hidden_layer_delta2)
    #print("weights_2_update:")
    #print(weights_2_update)
    weights_1_update = np.dot(x_train.T, hidden_layer_delta1) / N

    weights_3 = weights_3 + (learning_rate * weights_3_update)
    weights_2 = weights_2 + (learning_rate * weights_2_update)
    weights_1 = weights_1 + learning_rate * weights_1_update


monitoring_df = pd.DataFrame(monitoring)

fig, axes = plt.subplots(1, 2, figsize=(15,5))

monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error")


print("Neural Network (2 hidden layers) Evaluation:")
print("Mean Absolute Error: " + str(monitoring_df["mean_absolute_error"].min()))
print("Mean Squared Error: " + str(monitoring_df['mean_squared_error'].min()))
print("Root Mean Squared Error: " + str(monitoring_df["root_mean_squared_error"].min()))
plt.show()
