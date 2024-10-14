import math

file = open("input.txt", "r")
matrixA = []
vectorB = []
while True:
    aux_vector = []
    line = file.readline()
    if not line:
        break
    terms = line.split()
    limit = 2
    if terms[-2] == "-":
        vectorB.append(-int(terms[-1]))
        limit = 3
    else:
        vectorB.append(int(terms[-1]))
    sign = 1
    for term in terms[:-limit]:
        if term[0].isdigit():
            number = int(term[0])
            k = 1
            while term[k].isdigit():
                number = number * 10 + int(term[k])
                k += 1
            if sign:
                aux_vector.append(int(number))
            else:
                aux_vector.append(-int(number))
        elif term[0].isalpha():
            if sign:
                aux_vector.append(1)
            else:
                aux_vector.append(-1)
        elif term[0] == "-":
            sign = 0
        elif term[0] == "+":
            sign = 1
    matrixA.append(aux_vector)

def determinant(matrix):
    det = 0
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det += matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    return det

def trace(matrix):
    trace = 0
    for i in range(len(matrix)):
        trace += matrix[i][i]
    return trace

def norm(vector):
    norm = 0
    for element in vector:
        norm += element ** 2
    return math.sqrt(norm)

def transpose(matrix):
    matrixT = []
    for i in range(len(matrix[0])):
        aux_vector = []
        for j in range(len(matrix)):
            aux_vector.append(matrix[j][i])
        matrixT.append(aux_vector)
    return matrixT

def mul(matrix, vector):
    result = []
    for i in range(len(matrix)):
        aux = 0
        for j in range(len(vector)):
            aux += matrix[i][j] * vector[j]
        result.append(aux)
    return result

def copy_matrix(matrix):
    copy = []
    for row in matrix:
        copy.append(row.copy())
    return copy

def Cramer(matrix, vector):
    detA = determinant(matrix)
    if detA == 0:
        return "The system has either no solution or infinite solutions"

    # find x
    Ax = copy_matrix(matrix)
    for i in range(len(vector)):
        Ax[i][0] = vector[i]
    detAx = determinant(Ax)
    x = detAx / detA
    
    # find y
    Ay = copy_matrix(matrix)
    for i in range(len(vector)):
        Ay[i][1] = vector[i]
    detAy = determinant(Ay)
    y = detAy / detA
    
    # find z
    Az = copy_matrix(matrix)
    for i in range(len(vector)):
        Az[i][2] = vector[i]
    detAz = determinant(Az)
    z = detAz / detA
    
    return "Solution using Cramer's rule:\nx = " + str(x) + "\ny = " + str(y) + "\nz = " + str(z)

def cofactor_matrix(matrix):
    cofactor_matrix = []
    for i in range(len(matrix)):
        aux_row = []
        for j in range(len(matrix)):
            aux = (-1)**(i+j) * determinant([row[:j] + row[j+1:] for row in matrix[:i] + matrix[i+1:]])
            aux_row.append(aux)
        cofactor_matrix.append(aux_row)
    return cofactor_matrix

def adjugate_matrix(matrix):
    return transpose(cofactor_matrix(matrix))

def inverse_matrix(matrix):
    det = determinant(matrix)
    adj = adjugate_matrix(matrix)
    result = []
    if det == 0:
        return "Matrix is not invertible"
    else:
        for i in range(len(adj)):
            aux_row = []
            for j in range(len(adj)):
                aux_row.append(adj[i][j] / det)
            result.append(aux_row)
        return result

def solve_using_inversion(matrix, vector):
    result = mul(inverse_matrix(matrix), vector)
    return "Solution using inversion:\nx = " + str(result[0]) + "\ny = " + str(result[1]) + "\nz = " + str(result[2])

print(Cramer(matrixA, vectorB))
print()
print(solve_using_inversion(matrixA, vectorB))
print()
print(matrixA)
print(vectorB)

# Bonus
# Yes, there are similarities between the cofactor and the provided formula for the determinant of the matrix A.
# The formula for the determinant of a 3x3 matrix can bee seen as a sum of the products of the elements of the first row with the determinant of the 2x2 minors (the matrix that remains after removing the row and column of the element) and with the sign that alternates (-1)**(i+j).
# Thus, det(A) can be written as: det(A) = a11 * Cof(a11) + a12 * Cof(a12) + a13 * Cof(a13) = (-1)**(1+1) * a11 * (a22 * a33 - a23 * a32) + (-1)**(1+2) * a12 * (a21 * a33 - a23 * a31) + (-1)**(1+3) * a13 * (a21 * a32 - a22 * a31) = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31).
