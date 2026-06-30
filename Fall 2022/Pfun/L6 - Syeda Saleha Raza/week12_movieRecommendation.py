# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 22:37:30 2022

@author: saleha.raza
"""
users = ['Ali','Adnan','Tariq']

movies = ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10']

ratings = [[1,1,1,0,0,1,0,0,0,-1],
       [1,1,1,0,1,0,-1,0,1,-1],
       [1,0,-1,0,1,-1,0,0,0,-1],
      ]


#find similarity between users based on number of mmutually liked movies
def similarity(u1,u2):
    m1 = ratings[users.index(u1)]
    m2 = ratings[users.index(u2)]
    
    common = 0
    for i in range(len(m1)):
        if m1[i] != 0 and m2[i]!= 0 and m1[i] == m2[i]:
            common += 1
            
    return common    

#Which user is most similar to the given user
def mostSimilar(u1):
    closestUser = -1
    maxSimilarity = -1
    for u in users:
        if u != u1:
            sim = similarity(u1, u)
            if sim > maxSimilarity:
                maxSimilarity = sim
                closestUser = u
    return closestUser


def recommend(u1):
    #find most similar user
    closestUser = mostSimilar(u1)
    
    #Find movies that the closest user has liked and u1 has not rated
    m1 = ratings[users.index(u1)]
    m2 = ratings[users.index(closestUser)]
    
    recommended = []
    for i in range(len(m2)):
        if m2[i] == 1 and m1[i] == 0:
            recommended.append(movies[i])

    return recommended
            
print(similarity('Ali','Adnan'))
print(similarity('Adnan','Tariq'))
print(similarity('Ali','Tariq'))

print('Most similar to Adnan:', mostSimilar('Adnan'))        
print('Most similar to Ali:', mostSimilar('Ali'))
print('Most similar to Tariq:', mostSimilar('Tariq'))


print(recommend('Ali'))
print(recommend('Adnan'))
print(recommend('Tariq'))