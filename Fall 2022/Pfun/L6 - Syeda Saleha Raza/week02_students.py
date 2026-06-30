# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:40:57 2022

@author: saleha.raza
"""
S1 = 83
S2 = 92.4
S3 = 77.6
S4 = 61.4
S5 = 43.2

N1 = 'Maria'
N2 = 'Asad'
N3 = 'Bushra'
N4 = 'Fahad'
N5 = 'Faraz'

"""
#Taking User's input
N1 = input('Please enter name of S1:')
N2 = input('Please enter name of S2:')
N3 = input('Please enter name of S3:')
N4 = input('Please enter name of S4:')
N5 = input('Please enter name of S5:')

S1 = float(input('Please enter marks of S1:'))
S2 = float(input('Please enter marks of S2:'))
S3 = float(input('Please enter marks of S3:'))
S4 = float(input('Please enter marks of S4:'))
S5 = float(input('Please enter marks of S5:'))
"""
print("\"Name\"\t\"Marks\"")
print(N1,'\t',S1)
print(N2,'\t',S2)
print(N3,'\t',S3)
print(N4,'\t',S4)
print(N5,'\t',S5)

# String formatting in print

avg = (S1+S2+S3+S4+S5)/5
print('Average of this class is ', avg)

#Overriding seperator (default is space)
print (S1,S2,S3,S4,S5,sep='\n')

#More escape charaters (\n, \t, \", \',\\)

#Compute grade of S1
score = S4
grade = 'TBC'
message =''

if score >93:
    grade = 'A+'
    message ='Excellent, you are upto the mark!'
elif score >80:
    grade = 'A'
    message ='Good Job,Keep it up!'
elif score >70:
    grade = 'B'
    message = 'Can do better!'
elif score >60:
    grade = 'C'
    message = 'pata tha yahi ho ga'
else: 
    grade = 'F'
    message ='aatey hi kyon ho'

print(grade +"\n"+message)


