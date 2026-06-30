# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:39:08 2022

@author: saleha.raza
"""

import turtle
s=turtle.getscreen()
t = turtle.Turtle()
s.reset()

def gotoxy(t,x,y):
    t.penup()
    t.goto(x,y)
    t.pendown()
    

# playing wiht hexagons
# x,y = 0,0
# colors = ['purple','green','pink','blue','yellow','red']
# t.clear()
# length = 100
# for c in range(5):
#     gotoxy(t,x,y)
#     t.pencolor(colors[c])
#     for i in range(6):
#         t.left(60)
#         t.forward(length)
#     length = length - 15
#     x += 15
#     y += 7.5
    
    
# Playering with circle    
colors = ['purple','green','pink','blue','yellow','red']
x,y = -100,-100
t.clear()
for i in range(5):
    radius = 50
    c = 0
    gotoxy(t,x,y)
    while radius > 10:
        t.pencolor(colors[c%6])
        gotoxy(t,x,y)
        t.circle(radius)
        radius = radius - 10
        c += 1
        x = x+10
    y = y +20
