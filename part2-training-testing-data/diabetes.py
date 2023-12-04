import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
y = data.target.values
data = data.frame
# print(data)
x = data["bmi"].values

x = x.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .10)

# Create the model
model = LinearRegression().fit(xtrain, ytrain)

# get the coef_, intercept_ valuesm and r^2 values
# use float() to turn the arrays into a single float value
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

# print out the linear equation and r^2 value
print("Model's Linear Equation: y=",coef, "x+", intercept)
print("R Squared value:", r_squared)

'''
**********TEST THE MODEL**********
'''

# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)

# compare the actual and predicted values
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)

# Graph the data
# sets the size of the graph
plt.figure(figsize=(6, 4))

# creates a scatter plot and labels the axes
plt.scatter(xtrain,ytrain, c="blue")
plt.scatter(xtest, ytest, c="red")
plt.plot(xtrain, coef*xtrain + intercept, c="r", label="Line of Best Fit")

plt.xlabel("bmi")
plt.ylabel("quantitative measure")
plt.title("quantitative measure vs bmi")

# show the plot
plt.legend()
plt.show()