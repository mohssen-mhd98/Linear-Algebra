import numpy as np
np.set_printoptions(linewidth=300)


def gets_input():  # Takes elements of row for a matrix
    i = 0
    while i < row_num:
        st = input("Enter row{}:\n".format(i + 1))
        tmp = st.split()
        if tmp.__len__() != col_num:
            print("You have entered Wrong number of elements for this row, please try again")
            i = i - 1
        matrix[i] = matrix[i] + list(map(float, tmp))
        i += 1


def zeros_bottom():  # Interchange rows with all zeros element whit non-zero rows in a matrix
    val = row_num - 1  # Lowest row of matrix in each iteration
    # result = np.all((matrix == 0), axis=1)
    for r in range(row_num):
        row = matrix[r]
        c = 0
        count = 0
        while c < col_num and val > r:
            if row[c] == 0:
                count += 1
            if count == col_num:
                while not np.any(matrix[val, :col_num]):
                    val -= 1
                matrix[[r, val]] = matrix[[val, r]]
            c += 1
        # print("---\n", matrix)


def make_echelonform(): # Makes echelon form of a matrix
    val = 0  # Top row of matrix in each iteration
    for c in range(0, col_num):
        column = matrix[:, c]
        r = val
        flag = False
        while r < row_num and val < row_num:
            if column[r] != 0:
                if r > val:
                    matrix[[val, r]] = matrix[[r, val]]
                pps.append([val, c])
                val += 1
                flag = True
                break
            r += 1

        if flag:
            for i in range(val, row_num):
                matrix[i] += matrix[val - 1] * (-matrix[i, c] / matrix[val - 1, c])
                for j in range(0, col_num + 1):
                    matrix[i, j] = round(matrix[i, j], 2)
        # print("---\n", np.matrix(matrix))


def make_reduced_echelonform(length):  # makes reduced echelon form of a matrix
    for c in range(length):
        pp = pps[length - 1]

        matrix[pp[0]] /= matrix[pp[0], pp[1]]
        for i in range(0, pp[0]):
            matrix[i] += -1 * matrix[pp[0]] * matrix[i, pp[1]]
            for j in range(0, col_num + 1):
                matrix[i, j] = round(matrix[i, j], 2)
        length -= 1

        for i in range(row_num):
            for j in range(0, col_num + 1):
                matrix[i, j] = round(matrix[i, j], 2)


def describe_parametric():  # Shows equivalent system equations of a matrix if exist an answer
    free_var = ""  # Free variables of system equations
    sys_eq = ""  # System equations
    for i in range(row_num):
        lhs = ""
        rhs = ""
        for j in range(0, col_num):
            if j in [k[1] for k in pps]:
                if matrix[i, j] != 0:
                    if lhs == "":
                        lhs = str(matrix[i, j]) + "X" + str(j + 1)
                    else:
                        lhs += " + " + str(matrix[i, j]) + "X" + str(j + 1)
            else:
                if i == 0:
                    if free_var == "":
                        free_var = "X" + str(j + 1)
                    else:
                        free_var += "," + "X" + str(j + 1)
                if matrix[i, j] != 0:
                    if rhs == "":
                        rhs = str(-1 * matrix[i, j]) + "X" + str(j + 1)
                    else:
                        if matrix[i, j] * -1 < 0:
                            rhs += str(-1 * matrix[i, j]) + "X" + str(j + 1)
                        else:
                            rhs += " + " + str(-1 * matrix[i, j]) + "X" + str(j + 1)
            if j + 1 == col_num and lhs != "":
                if matrix[i, j + 1] < 0:  # if const value is negative
                    rhs += str(matrix[i, j + 1])
                else:
                    if rhs == "":
                        rhs = str(matrix[i, j + 1])
                    else:
                        rhs += " + " + str(matrix[i, j + 1])
            # elif j+1 == col_num and lhs == "" and rhs == "":
            #     rhs = 0
            #     lhs = 0
        if not np.any(matrix[i, :j + 1]) and matrix[i, j + 1] != 0:
            print("Equation system has no answer!")
            sys_eq = ""
            break
        # print(lhs, " = ", rhs)
        if sys_eq == "":
            sys_eq = lhs + " = " + rhs + "\n"
        elif lhs != "" and rhs != "":
            sys_eq += lhs + " = " + rhs + "\n"

    return (sys_eq, free_var)


st = input("Enter number of rows and columns Respectively:\n")
rc = st.split()
row_num = int(rc[0])
col_num = int(rc[1])
# print(column, row)
matrix = np.zeros([row_num, col_num])

'''
matrix = np.array([[0, 6, 82, 0, 30, 0, 0],
                   [17, 20, 8, 11, 45, 34, 19],
                   [34, 40, 16, 22, 20, 68, 28],
                   [0, 19, 90, 9, 12, 0, 23],
                   [14, 16, 45, 0, 0, 0, 21],
                   [0, 6, 82, 0, 30, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
matrix = np.array([[0, 0, 1, -2],
                  [1, -7, 0, 6],
                  [-1, 7, -4, 2]])
print(const_val.reshape(row, 1))
'''
# Gets elements of each row
gets_input()

st = input("Enter constant values:\n")
const_val = st.split()  # constant values
const_val = np.array(list(map(float, const_val)))

# print(const_val.reshape(row_num, 1))

# Add constant values as last column in matrix
matrix = np.hstack((matrix, const_val.reshape(-1, 1)))
print("Given matrix:\n", matrix)

pps = []  # Pivot Positions of matrix (i, j)

# Move rows that are all zeros to bottom of matrix
zeros_bottom()

# Gives echelon form equivalent of matrix
make_echelonform()

length = len(pps)

# Gives reduced echelon form of matrix
make_reduced_echelonform(length)

print("Matrix in Reduced echelon form:\n", matrix)

# Returns parametric description of matrix
sys_eq, free_var = describe_parametric()

if sys_eq != "":
    print("System of linear equations:")
    print(sys_eq)
    print("Free variables are:", free_var)
