# reference: https://web.stanford.edu/class/cs231a/section/section1.pdf
# Lists
empty_list = list()
also_empty_list = []
zeros_list = [0]*5
print(zeros_list)
number_list = [16]*5
print(number_list)

print(empty_list)
empty_list.append(1)
print(empty_list)
print(len(empty_list))

# List indexing
list_var = range(10)
print(list_var)
print(list_var[4])  
print(list_var[4:7])
print(list_var[0::3])

print(list_var)
print(list_var[-1])
print(list_var[::-1])
print(list_var[3:1:-1])

# Dictionaries 
empty_dict = dict()
also_empty_dict = {}
filled_dict = {3:'Hello,',4:'World!'}
print(filled_dict)
print(filled_dict[3]+filled_dict[4])

filled_dict[5]=' Me '
filled_dict[6]='machine'
print(filled_dict)
print(filled_dict[3]+filled_dict[4]+filled_dict[5]+filled_dict[6])

del filled_dict[3]
print(filled_dict)
print(len(filled_dict))
print("")

# Functions
def add_numbers(a,b):
    return a+b
 
print(add_numbers(2,3))

lambda_add_number = lambda a,b: a+b
print(lambda_add_number(4,6))

# Loops, List and Dictionary Comprehensions
for i in range(10):
    print('Looping %d' % i)

filled_list=[a/2 for a in range(10)]
print(filled_list)
filled_dict={a:a**2 for a in range(5)}
print(filled_dict)

# Matrices and vectors
print("# Matrices and vectors")
print("")

import numpy as np

M=np.array([[1,2,3],
           [4,5,6],
           [7,8,9]])
           
v=np.array([[1],
            [2],
            [3]])

print(M.shape)
print(v.shape)

v_single_dim = np.array([1,2,3])
print(v_single_dim.shape)

print(v+v)
print(" ")
print(3*v)
print(" ")

# other ways of creating matrices and vectors

a=np.zeros((2,2)) # 2x2 zero matrix
print(a)
print("")

b=np.ones((1,2))  # 1X2 unit matrix
print(b)
print("")

c=np.full((2,2),7)# 2x2 matrix filled with 7
print(c)
print("")

d=np.eye(2)       # 2x2 identity matrix
print(d)
print("")

e=np.random.random((2,2)) #2x2 matrix of random numbers
print(e)
print("")

v1=np.array([1,2,3])
v2=np.array([4,5,6])
v3=np.array([7,8,9])
M=np.vstack([v1,v2,v3])
print(M)
print("")
c1=np.array([[1],
             [2],
             [3]])
c2=np.array([[4],
             [5],
             [6]])
c3=np.array([[7],
             [8],
             [9]])
N=np.hstack([c1,c2,c3])
print(N)
print("")

# Matrix indexing
print("# Matrix indexing")
print("")

print(M[:2,1:3])
print("")
print(N[0,2])

# M Template
# __            __
# | [  ,   ,   ] |
# | [  ,   ,   ] |
# | [  ,   ,   ] |
# | [  ,   ,   ] |
# --            --
c1=np.array([[],
             [],
             []])

# Dot product of matrices and vectors
print("# Dot product of matrices and vectors")
print("")

c1=np.array([[3],
             [2],
             [0]])
c2=np.array([[0],
             [0],
             [1]])
c3=np.array([[2],
             [-2],
             [1]])

M=np.hstack([c1,c2,c3])
v=np.array([[1],
            [2],
            [3]])
print(M)
print(v)
print("")

print(M.dot(v))  # Dot product
print("")
print(v.T.dot(v))
print("")

# Cross product of matrices and vectors
print("# Cross product of matrices and vectors")
print("")

v1=np.array([[3],
             [-3],
             [1]])
v2=np.array([[4],
             [9],
             [2]])
             
print(np.cross(v1,v2,axisa=0,axisb=0).T) # Slightly convoluted because np.cross() assumes
print("")

# Element wise multiplication
print("# Element wise multiplication")
print("")

print(M)
print("")
print(v)
print("")
print(np.multiply(M,v))
print("")
print(np.multiply(v,v))
print("")

# Transpose of a matrix
print("# Transpose of a matrix")
print("")

print(M.T)
print("")
print(v.T)
print("")
print(M.shape)
print(M.T.shape)
print("")
print(v.shape)
print(v.T.shape)
print("")

# Determinant of a matrix
print("# Determinant of a matrix")
print("")

print(np.linalg.det(M))
print("")

# Inverse of a matrix
print("# Inverse of a matrix")
print("")

print(np.linalg.inv(M))
print("")

# Eigenvalues and eigenvectors of a matrix
print("# Eigenvalues and eigenvectors of a matrix")
print("")

c1=np.array([[0],
             [-2]])

c2=np.array([[1],
             [-3]])
M=np.hstack([c1,c2])
print("Matrix")
print(M)
print("")

eigvals,eigvecs = np.linalg.eig(M)
print("Eigenvalues")
print(eigvals)
print("")
print("Eigenvectors")
print(eigvecs)
print("")

# Singular Value Decomposition
print("# Singular Value Decomposition")
print("")

c1=np.array([[3],
             [2],
             [0]])
c2=np.array([[0],
             [0],
             [1]])
c3=np.array([[2],
             [-2],
             [1]])

M=np.hstack([c1,c2,c3])
print("Matrix")
print(M)
print("")
U,S,Vtranspose = np.linalg.svd(M)
print("U")
print(U)
print("")
print("S")
print(S)
print("")
print("Vtranspose")
print(Vtranspose)
print("")