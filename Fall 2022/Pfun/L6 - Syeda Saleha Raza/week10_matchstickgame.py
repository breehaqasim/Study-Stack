# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:56:35 2022

@author: saleha.raza
"""

import random
sticks = 21
options = [1,2,3,4]
myturn = True

while sticks > 0:
    if (myturn):
        while (True):               #This while(True) + break can be replaced by maintaining a boolean variable as demonstrated. I don't like using break!
            i = int(input('How many sticks do you want to choose (1/2/3/4)?'))
            if i in options:
                break
          
            print("Please enter a number between 1 and 4.")
   
    else:       #computer's turn
        if (sticks <= 5):
            i = max(sticks -1,1)
        elif sticks <= 10:
            i = max(sticks - 6,1)
        else:        
            i = random.choice(options)
        print('The computer has choosen',i,'sticks. Its your turn now:')
    
    sticks = sticks - i
    print('Sticks left', sticks)    

    if sticks > 0:
        myturn = not myturn    #This is similar to 'If myturn is True then make it False and vice versa'.
   

if myturn:
    print ('You have lost the game.')
else:
    print('You have won the game.')           