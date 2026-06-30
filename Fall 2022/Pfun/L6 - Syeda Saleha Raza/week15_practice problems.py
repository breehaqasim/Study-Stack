# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:32:22 2022

@author: saleha.raza
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:24:14 2022

@author: saleha.raza
"""


# #Chatroom
# inp = 'olehelolo'

# target = 'hello'

# i = 0
# pos = -1

# found = True
# while pos < len(inp) and i < len(target):
#     pos = inp.find(target[i],pos+1) #inp.index(target[i],pos)
#     if pos == -1:
#        found = False
#        break
#     i += 1

# if found:    
#     print('YES')
# else:
#     print('NO')


#-----------------------------------------------------------------------------
#Max Pooling
lst = [[1,1,2,4],
       [5,6,7,8],
       [3,2,1,0],
       [1,2,3,4]]

pooled = []
row = 0
for i in range(0,len(lst),2):
    pooled.append([])
    for j in range(0,len(lst[0]),2):
        print(lst[i][j], lst[i+1][j+1])
        s = max(lst[i][j],lst[i+1][j],lst[i][j+1],lst[i+1][j+1])
        pooled[row].append(s)

    row +=1
print(pooled)

#----------------------------------------------------------------------
nos = [2,3,4,5,6,7,8,9]
def RecursiveSum(A):
    print(A)
    if len(A) == 1:
        return A[0]
    
    B= []
    for i in range(len(A)//2):
        B.append(A[2*i] + A[2*i +1])
    
    return RecursiveSum(B)

print(RecursiveSum(nos))    
#---------------------------------------------------------------------
#Puzzle pieces - min difference
# n = int(input().split(' ')[0])
# inputlist = input().split(' ')

# pieces = []
# for i in inputlist:
#     pieces.append(int(i))

# pieces = sorted(pieces)

# mindiff = max(pieces)
# for i in range(len(pieces)):
#     subset = []
#     for j in range(i,i+4):
#         subset.append(pieces[j])
#     diff = max(subset) - min(subset)
    
#     if diff < mindiff:
#         mindiff = diff

# print(mindiff)
#-----------------------------------------------------------
#Matrix Multiplication
# mat1 = [ [3,9,8,4], [4,5,6,4],[1,3,4,4],[1,2,3,4]]
# mat2 = [ [1,4,7,4], [5,3,2,4],[6,7,8,4],[4,3,2,1]]

# def matrixMultiplication(r,c):
#     n = len(mat1[r])
#     result = 0    
#     for i in range(n):
#         result +=  mat1[r][i] * mat2[i][c]
        
#     return result    
            

# m = len(mat1)
# n = len(mat1[0])


# mat3 = []
# print(mat3)
# for r in range(m):
#     mat3.append([])
#     for c in range(n):
#         print(r,c)
#         mat3[r].append(multRowCol(r, c))
#         print(mat3)
# print(mat3)   