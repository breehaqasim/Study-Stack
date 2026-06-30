# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:17:27 2022

@author: saleha.raza
"""

"""
There are three players on the soccer field p1, p2, p3. They need to collaborate and take the ball towards the opponent goal. The opponent players
o1,o2,o3 are there to counter their moves and attack home goal. The position of ball and the home and opp goal are also given.

Let's write code to decide on:
who should become attacker?
who should become defender?
who should become forward?

how many players (of yours and opponent's team) are in your half?
how many players are in the opponent's half?
how many players are in the penalty area?
Is the ball being crowded?
Are you under attack?
Which action should the attacker perform (kick, dribble or give pass to a nearby teammate)?

"""
import math
#field boundaries
fieldx_min,fieldx_max  = -10,10
fieldy_min, fieldy_max = -20, 20

#home goal position
homegoalx,homegoaly = 0, -20

#opponent goal position
oppgoalx, oppgoaly = 0, 20

#players' positions
p1x,p1y = 3.4,5.6
p2x,p2y = -1.4,2.9
p3x,p3y = 0.5,8.3

#Opponent players' positions
o1x, o1y = 3.4,4.9
o2x, o1y = -0.5,5.6
o3x, o3y = -3.9,0.8

#ball position
ballx, bally = -2.4,1.1

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 +(y2-y1)**2)

def inside_box(px,py, bx_min,bx_max, by_min, by_max):
    if px >= bx_min and px <= bx_max and py >= by_min and py <= by_max:
        return True
    return False

def inside_field(px,py):
    return inside_box(px,py,fieldx_min,fieldx_max,fieldy_min,fieldy_max)
    
p1ball = distance(p1x,p1y,ballx,bally)
p2ball = distance(p2x,p2y,ballx,bally)
p3ball = distance(p3x,p3y,ballx,bally)

#who should become attacker?
#the players closest to the ball is attacker
attacker = -1
if p1ball < p2ball and p1ball < p3ball:
    attacker = 1
elif p2ball < p1ball and p2ball < p3ball:
   attacker = 2
else:
   attacker = 3
   
#who should become defender?
#the player(other than attacker) closest to the home goal is defender     
p1Goal = distance(p1x, p1y, homegoalx, homegoaly)
p2Goal = distance(p1x, p1y, homegoalx, homegoaly)
p3Goal = distance(p1x, p1y, homegoalx, homegoaly)

defender = -1
if (p1Goal < p2Goal) and (p1Goal < p3Goal) and attacker != 1:
    defender = 1
elif (p2Goal < p1Goal) and (p2Goal < p3Goal) and attacker != 2:
    defender = 2
else:
    defender = 3
    
#the third player is forward         
if attacker != 1 and defender != 1:
    forward = 1
elif attacker != 2 and defender != 2:
    forward = 2
else:
    forward = 3
    

count_outside = 0

if not (inside_field(p1x,p1y)):
    count_outside = count_outside +1


#to be continued in the next class