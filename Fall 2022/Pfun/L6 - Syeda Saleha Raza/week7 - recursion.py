# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:53:59 2022

@author: saleha.raza
"""

def factorial(n):
    if n==1:
        return 1
    return n*factorial(n-1)

print(factorial(5))


def power(k,n):
    if (n==1):
        return k
    return k* power(k,n-1)


print(power(5,6),pow(5,6))


def countdown(n):
    print(n)
    if (n>1):
        countdown(n-1)
    
countdown(15)   


def countup(n):
    if (n>=1):
        countup(n-1)
        print(n)
    
countup(15)   

def clap(k):
    print('clap...',k)
    if (k>0):
        clap(k-1)
                
clap(10)

def fibonacci(n):
    if (n==1):
        return 0
    elif n==2:
        return 1
    
    return fibonacci(n-1) + fibonacci(n-2)

def isPalindrome(s):
    print(s,len(s),s[0],s[-1])
    
    if len(s) == 1:
        return True
        
    if s[0].lower() == s[-1].lower():
        return isPalindrome(s[1:-1].strip(' '))
    else: return False
    

print(isPalindrome("was it a car or a cat I saw"))