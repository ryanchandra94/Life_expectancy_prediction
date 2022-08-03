import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import scipy


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




###### 1. Linear Regression
######## 1.1 Simple Linear Regression

linear_train_x, linear_test_x, linear_train_y, linear_test_y = train_test_split(df_clean, df_target,
                                                    test_size=0.20, random_state=1)


simple_models = []
simple_models.append(("Adult_Mortality", LinearRegression()))
simple_models.append(("infant_deaths", LinearRegression()))
simple_models.append(("Alcohol", LinearRegression()))
simple_models.append(("percentage_expenditure", LinearRegression()))
simple_models.append(("Hepatitis_B", LinearRegression()))
simple_models.append(("Measles", LinearRegression()))
simple_models.append(("BMI", LinearRegression()))
simple_models.append(("under_five_deaths", LinearRegression()))
simple_models.append(("Polio", LinearRegression()))
simple_models.append(("Total_expenditure", LinearRegression()))
simple_models.append(("Diphtheria", LinearRegression()))
simple_models.append(("HIV/AIDS", LinearRegression()))
simple_models.append(("GDP", LinearRegression()))
simple_models.append(("Population", LinearRegression()))
simple_models.append(("thinness  1-19 years", LinearRegression()))
simple_models.append(("thinness 5-9 years", LinearRegression()))
simple_models.append(("Income_composition_of_resources", LinearRegression()))
simple_models.append(("Schooling", LinearRegression()))

simple_lr_results = []
independent_variable = []

for independent, linear in simple_models:
    independentVar = linear_train_x[independent].values.reshape(-1,1)
    dependentVar = linear_train_y.values
    train_simple = cross_val_score(linear, independentVar, dependentVar,
                                   scoring='r2')
    simple_lr_results.append((independent, train_simple.mean()))


# Draw 1 example of fit line (Alcohol, life_expectancy)
x_plot = df_clean["Schooling"].values.reshape(-1,1)
y_plot = df_target
plt.scatter(x_plot, y_plot)

model = LinearRegression().fit(x_plot, y_plot)
plt.plot(x_plot, model.predict(x_plot), color="red")
plt.title("Relationship between Schooling and life expectancy")
plt.xlabel('Schooling')
plt.ylabel('Life Expectancy')
plt.show()
print("R-squared between each independent variable and life expectancy" + str(simple_lr_results))

# Divide independent variable into 5 categories
# Immunization, mortality, economic, social, and others
# lets test the relationship between these categories with life expectancy

immunization = ["Polio", "Diphtheria", "Measles", "Hepatitis_B"]
mortality = ["Adult_Mortality", "under_five_deaths", "infant_deaths"]
economic = ["percentage_expenditure", "Total_expenditure", "GDP", "Income_composition_of_resources"]
social = ["Population", "Schooling", "Alcohol"]
other = ["BMI", "HIV/AIDS", "thinness  1-19 years", "thinness 5-9 years"]


model_list = []
model_list.append(("immunization", immunization))
model_list.append(("mortality", mortality))
model_list.append(("economic", economic))
model_list.append(("social", social))
model_list.append(("other", other))
result_list = []

for a, b in model_list:
    x = linear_train_x[b].values
    y = linear_train_y
    train = cross_val_score(LinearRegression(), x, y, scoring='r2')
    result_list.append((a, train.mean()))
print("R-square between categories and life expectancy", result_list)

# Multi Linear Regression Score
#Using cross validation score to check linear regression result

linear_models = []
linear_models.append(("Linear Regression", LinearRegression()))
##models.append(("Linear Discrimination Analysis", LinearDiscriminantAnalysis()))

lr_results = []

lr_name = []

for name, model in linear_models:
    kfold = KFold(n_splits=10, shuffle=False)
    linear_results = cross_val_score(model, linear_train_x,
                                 linear_train_y, cv=kfold, scoring='r2')
    lr_results.append(linear_results)

    lr_name.append(name)

print("%s: %f (%f)" % (name, linear_results.mean(), linear_results.std()))



## Create new column to store classification of life expectancy
## life expectancy >= mean -> 1
## life expectancy < mean -> 0
df['New Column'] = np.where(df['Life_expectancy'] >= target_mean, 1, 0)
cl_target = df['New Column']
cl_training_x, cl_test_x, cl_training_y, cl_test_y = \
    train_test_split(df_clean, cl_target, test_size=0.20, random_state=1)


## Divide data for classification algorithm
## I compare several classification algorithm
## Logistic regression, Linear discrimination analysis,
## KNeighborsClassifier, DecisionTreeClassifier,
## GaussianNB, SVC


cl_models = []
cl_models.append(('LogR', LogisticRegression()))
cl_models.append(('LDA', LinearDiscriminantAnalysis()))
cl_models.append(('KNN', KNeighborsClassifier()))
cl_models.append(('DT', DecisionTreeClassifier()))
cl_models.append(('RF', RandomForestClassifier()))
cl_models.append(('NB', GaussianNB()))
cl_models.append(('SVM', SVC()))

cl_results = []

cl_names = []

scoring = 'accuracy'

for cl_name, cl_model in cl_models:
    cl_kfold = KFold(n_splits=10, shuffle=False)
    cv_cl_result = cross_val_score(cl_model,
                                   cl_training_x, cl_training_y, cv=cl_kfold,
                                   scoring=scoring)
    cl_results.append(cv_cl_result)
    cl_names.append(cl_name)
    print("%s: %f (%f)" % (cl_name, cv_cl_result.mean(), cv_cl_result.std()))



# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Comparison between different classificaton algorithm')
ax = fig.add_subplot(111)
plt.boxplot(cl_results)
ax.set_xticklabels(cl_names)
##plt.show()


#### Lets test simple linear regression, multi linear regression, decision tree and random forest
### decision tree and random forest have better accuracy than others (test accuracy)
### also we compare the error rate using (MAE, MSE, and RMSE) of simple and multi linear regression


# Simple linear regression (Schooling vs Life expectancy)
x_variable = df_clean['Schooling'].values.reshape(-1,1)
y_variable = df_target
train_x, test_x, train_y, test_y = train_test_split(x_variable, y_variable, test_size=0.5, random_state=1)
lm = LinearRegression()

lm.fit(train_x, train_y)
y_pred = lm.predict(np.array(test_x).reshape(-1,1))
print("Simple Linear Regression: ")
print("Mean Absolute Error: " + str(mean_absolute_error(test_y, y_pred)))
print("Mean Squared Error: " + str(mean_squared_error(test_y, y_pred)))
print("Root Mean Squared Error: " + str(np.sqrt(mean_squared_error(test_y, y_pred))))
print("\n")

# Multi linear regression
train_x, test_x, train_y, test_y = train_test_split(df_clean, df_target, test_size=0.5, random_state=1)
lm.fit(train_x, train_y)
y_pred = lm.predict(test_x)
print("Multi Linear Regression: ")
print("Mean Absolute Error: " + str(mean_absolute_error(test_y, y_pred)))
print("Mean Squared Error: " + str(mean_squared_error(test_y, y_pred)))
print("Root Mean Squared Error: " + str(np.sqrt(mean_squared_error(test_y, y_pred))))
print("\n")


# Decision Tree test accuracy
train_x, test_x, train_y, test_y = train_test_split(df_clean, cl_target, test_size=0.5, random_state=1)
tree_model = DecisionTreeClassifier()
tree_model.fit(train_x,train_y)
y_predict = tree_model.predict(test_x)
DT_accuracy = accuracy_score(test_y, y_predict)
print("Decision Tree accuracy: " + str(DT_accuracy))
print(classification_report(test_y, y_predict))

# Random Forest test accuracy
rf_model = RandomForestClassifier()
rf_model.fit(train_x,train_y)
y_predict = rf_model.predict(test_x)
rf_accuracy = accuracy_score(test_y, y_predict)
print("Random Forest Accuracy: " + str(rf_accuracy))
print(classification_report(test_y, y_predict))

##### Next, I try to build neural network for prediction (real value) and classification


# predict life expectancy (classification)
# output >= mean -> 1
# output < mean -> 0

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

n_input = 18
n_hidden = 10
n_output = 2

np.random.seed(10)

weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden)) #(4,2)
weights_2 = np.random.normal(scale=0.5, size=(n_hidden, n_output)) #(2,3)

monitoring = {"mean_squared_error": [], "accuracy": []}


for epoch in range(epochs):
    #feedforward
    hidden_layer_inputs = np.dot(x_train,weights_1)
    hidden_layer_outputs = sigmoid(hidden_layer_inputs)

    output_layer_inputs = np.dot(hidden_layer_outputs, weights_2)
    output_layer_outputs = sigmoid(output_layer_inputs)



    #monitor training process
    mse = mean_squarred_error(output_layer_outputs, y_train)
    acc = accuracy(output_layer_outputs, y_train)

    monitoring["mean_squared_error"].append(mse)
    monitoring["accuracy"].append(acc)



    #backpropagation
    output_layer_error = output_layer_outputs - y_train
    output_layer_delta = output_layer_error * output_layer_outputs * (1-output_layer_outputs)

    hidden_layer_error = np.dot(output_layer_delta, weights_2.T)
    hidden_layer_delta = hidden_layer_error * hidden_layer_outputs * (1-hidden_layer_outputs)

    #weights update
    weights_2_update = np.dot(hidden_layer_outputs.T, output_layer_delta) / N
    weights_1_update = np.dot(x_train.T, hidden_layer_delta) / N


    weights_2 = weights_2 - learning_rate * weights_2_update
    weights_1 = weights_1 - learning_rate * weights_1_update

monitoring_df = pd.DataFrame(monitoring)

fig, axes = plt.subplots(1, 2, figsize=(15,5))

monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error")
monitoring_df.accuracy.plot(ax=axes[1], title="Accuracy")
print(monitoring_df)
plt.show()




