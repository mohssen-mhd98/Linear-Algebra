import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

NAME = "better_result"


def main():

    # Get data from csv file
    open_list, df = read_data()
    y_list = open_list.copy()
    t_list = []
    ones_list = []

    # Create a list that holds elements of open_list.
    y_arr = np.array(y_list).reshape(-1, 1)

    # Create an array of one with a length correspond to y's elements
    for i in range(len(y_list)):
        t_list.append(i)
        ones_list.append(1)

    # Matrix that contains t elements in first column and ones_list in second column.
    x_matrix = design_matrix(y_list, degree=1)

    # Transposing x_matrix for making SVD decomposition
    transposed_x_matrix = x_matrix.transpose()

    # (X)T * X matrix
    a_matrix = np.dot(transposed_x_matrix, x_matrix)

    # b is result of (X)T * y_vector
    b = np.dot(x_matrix.T, y_arr)

    # This is the least square solution of design matrix and y vector
    beta = np.linalg.solve(a_matrix, b)

    print("Linear Regression Residuals :")
    print_residuals(df, beta, degree=1)
    plot_design(open_list, x_matrix, beta, t_list)

# quadratic ------------------------------------------------------------------------------------------------------------

    # Matrix that contains t elements in first column and ones_list in second column.
    x_matrix = design_matrix(y_list, degree=2)

    # Transposing x_matrix for making SVD decomposition
    transposed_x_matrix = x_matrix.transpose()

    # b is result of (X)T * y_vector
    a_matrix = np.dot(transposed_x_matrix, x_matrix)

    # b is result of (X)T * y_vector
    b = np.dot(transposed_x_matrix, y_arr)

    # This is the least square solution of design matrix and y vector
    beta = np.linalg.solve(a_matrix, b)

    print("\nNon-Linear Regression Residuals :")
    print_residuals(df, beta, degree=2)
    plot_design(open_list, x_matrix, beta, t_list)
    print("So Non-Linear Regression has errors with less value and it is better!")

# Functions ------------------------------------------------------------------------------------------------------------


# Prints the difference between predicted y-values and observed y-values
def print_residuals(df, beta, degree):
    if degree == 1:
        open_last10_elements = df.tail(10).Open.to_list()
        index_last10_elements = df.tail(10).index.to_list()
        count = 0
        for i in index_last10_elements:
            print("day: ", i)
            # y = b0 + b1*t
            cal_value = beta[0][0] + beta[1][0] * i
            actual_val = open_last10_elements[count]
            error = cal_value - actual_val
            count += 1

            print("Calculated value = {}".format(cal_value))
            print("Actual value = {}".format(actual_val))
            print("Error = {}".format(error))

    elif degree == 2:
        open_last10_elements = df.tail(10).Open.to_list()
        index_last10_elements = df.tail(10).index.to_list()
        count = 0
        for i in index_last10_elements:
            print("day: ", i)
            # y = b0 + b1*t + b2*(t^2)
            cal_value = beta[0][0] + beta[1][0] * i + beta[2][0] * (i ** 2)
            actual_val = open_last10_elements[count]
            error = cal_value - actual_val
            count += 1

            print("Calculated value = {}".format(cal_value))
            print("Actual value = {}".format(actual_val))
            print("Error = {}".format(error))


# Makes a X matrix of equation X.ß = y
def design_matrix(y_list, degree):

    if degree == 1:
        x_matrix = np.ones((len(y_list), 2))
        for i in range(len(y_list)):
            x_matrix[i][1] = round(i, 2)
            x_matrix[i][0] = round(1, 2)

    elif degree == 2:
        x_matrix = np.ones((len(y_list), 3))
        for i in range(len(y_list)):
            x_matrix[i][0] = round(1, 2)
            x_matrix[i][1] = round(i, 2)
            x_matrix[i][2] = round(i ** 2, 2)

    return x_matrix


# Returns elements of a special column, except for the last 10 elements.
def read_data():
    df = pd.read_csv('GOOGL.csv')
    open_list = df.Open.to_list()
    open_list = open_list[:len(open_list) - 10]
    return open_list, df


# Shows the plot of real data and predicted data
def plot_design(open_list, design_matrix, betha, t_list):
    predicted_y = np.dot(design_matrix, betha)
    predicted_y = predicted_y.reshape(1, -1).tolist()
    predicted_y_val = [x for x in predicted_y[0]]

    plt.plot(t_list, predicted_y_val, color='blue',  label='Predicted Line')
    plt.scatter(range(len(open_list)), open_list, color='red', label='Open')
    plt.legend(loc='best')
    plt.savefig(NAME)
    plt.show()


if __name__ == "__main__":
    main()
