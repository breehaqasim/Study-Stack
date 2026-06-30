# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:40:57 2022

@author: saleha.raza

Continuing with the our previous grade example (from week 02), let's explore the use of functions!
"""

def computeGrade(score):
    #Compute grade of S1

    grade = 'TBC'
    message =''

    if score >=92.5:
        msg_if_bumped(score, 93)
        grade = 'A+'
        message ='Excellent, you are upto the mark!'
    elif score >=79.5:
        msg_if_bumped(score, 80)
        grade = 'A'
        message ='Good Job,Keep it up!'
    elif score >=69.5:
        msg_if_bumped(score, 70)
        grade = 'B'
        message = 'Can do better!'
    elif score >=59.5:
        msg_if_bumped(score, 60)
        grade = 'C'
        message = 'pata tha yahi ho ga'
    else: 
        grade = 'F'
        message ='aatey hi kyon ho'

    #print(grade +"\n"+message)
    return grade

def msg_if_bumped(score, boundary):
    if score <boundary:
        print("Your grade has been bumped up!") 
        
def compute_GP(grade):
   
    if grade == 'A' or grade == 'A+':
        return 4.0
    elif grade =='B':
        return 3.0
    elif grade == 'C':
        return 2.0
    else:
        return 0.0
    

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
#More escape charaters (\n, \t, \", \',\\)

g1 = computeGrade(S1)
g2 = computeGrade(S2)
g3 = computeGrade(S3)
g4 = computeGrade(S4)
g5 = computeGrade(S5)

# String formatting in print

print("===========================")
print("Name\tMarks\tGrade\tGP")
print("===========================")
print(N1,'\t',S1,'\t',g1,'\t',compute_GP(g1))
print(N2,'\t',S2,'\t',g2,'\t',compute_GP(g2))
print(N3,'\t',S3,'\t',g3,'\t',compute_GP(g3))
print(N4,'\t',S4,'\t',g4,'\t',compute_GP(g4))
print(N5,'\t',S5,'\t',g5,'\t',compute_GP(g5))
print("===========================")
avg = (S1+S2+S3+S4+S5)/5
print('The average score of this class is ', avg)


"""
More todo:
Whta is average GP?
How many Fs are there?
    
"""
