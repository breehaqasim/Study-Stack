# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:29:48 2022

@author: saleha.raza

Exercise:
HU library charges a fine for every book returned late. The fine is 50 Rs per day for first 5 days, 
increases to 80 Rs per day from day 6 to day 10, and further increases to 100 Rs per day from day 10 
till day 20. If you return the book after 20 days, your membership will be cancelled. 
The policy is different for faculty who can be late for a month and will be charged Rs 100 per day 
afterwards.

Write a program to accept the number of days the member is late to return a book and the type of member 
and display the fine and the appropriate message.
"""

days = int(input('ENter the no of days you are late in returning the book:'))
memberType = input('Are you a faculty (Yes/No)?')  

if memberType == 'Yes' or memberType == 'yes':
    isFaculty = True
else:
    isFaculty = False    


fine = 0
if isFaculty:  #same as if isFaculty === True
    if days > 30:
        fine = (days-30)*100
elif not isFaculty: #same as if isFaculty == False
    if days <=5:
        fine = days * 50
    elif days <=10:
        fine = (5 * 50) + (days-5)*80
    elif days <=20:
        fine = (5 * 50) + (5*80) + (days-10)*100
    elif days > 20:
        fine = (5 * 50) + (5*80) + (days-10)*100
        print('Sorry, we regret that your membership has been cancelled!')
    else:
        print('Incorrect number of days entered.')
else:
    print('Incorrect member type')  

print ('Your fine is :'+ str(fine) +'/=')

"""
More to do:
Validating input - what if days < 0??  what if user enters something other than yes/no?
Are if...else conditions exhaustive? Are you missing some case?
Test your code...Make sure to test ALL branches of your code. Test on boundary values.
"""
