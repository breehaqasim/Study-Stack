# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:32:15 2022

@author: saleha.raza
"""

A = [ [2,3,9],[4,8,6],[5,13,20]]

B = [ [1,7,12],[4,6,13],[4,7,50]]

def printMatrix(M):
    # for i in M:
    #     for j in i

    for i in range(len(M)):
        for j in range(len(M[i])):
            print(M[i][j], end='\t')
        print()

def add(M1,M2):
    result = []
    if len(M1) != len(M2):
        print('Matrices should be of same dimension.')

    else:
        for i in range(len(M1)):
            result.append([])
            for j in range(len(M1[i])):
                result[i].append(M1[i][j] + M2[i][j])
    return result            
        
def diagonalMatrix(M):
    result=[]
    for i in range(len(M)):
        result.append([])
        for j in range(len(M[i])):
            if i == j:
                result[i].append(M[i][j])
            else:
                result[i].append(0)
    return result            
    
        
print('A:')
printMatrix(A)
print('B:')
printMatrix(B)
print('A+B:')
sumofTwo = add(A,B)
printMatrix(sumofTwo)
print('Diagonal:')
diag = diagonalMatrix(A)
printMatrix(diag)


