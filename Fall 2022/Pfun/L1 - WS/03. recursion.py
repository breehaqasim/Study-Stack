'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Weeks 6 and 7, Fall 2022

Recursive functions.
'''

def sum_first_n(n):
    '''
    Returns the sum of the first n numbers.

    Parameters:
    n: we need to sum the numbers from 1 to n

    Return:
    Sum of numbers from 1 to n inclusive.
    '''
    if n == 1:
        return 1
    else:
        return sum_first_n(n-1) + n

    # 1 + 2 + 3 + 4 + ... + n
    # (1 + 2 + 3 + 4 + ... + n-1) + n
    

def count(n:int):
    '''
    Prints the numbers from 1 to n on separate lines.

    Parameters:
    n: we need to print the numbers from 1 to n

    Return:
    None
    '''
    if n < 1:
        print('Invalid input.')
    elif n == 1:
        print(1)
    else:
        count(n-1)
        print(n)

'''
pseudocode

count(n)  # prints numbers from 1 to n
1. if n is 1, print 1, end
2. print numbers from 1 to (n-1)  # count(n-1)
3. print n
'''

def factorial(n):
    '''
    Returns n!

    Parameters:
    n: we need to compute n!

    Return:
    n!
    '''
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

'''
factorial

4! = 4 . 3. 2. 1 = 24
n! = n . (n-1) . (n-2) . ... . 1

factorial(n)  # returns the factorial of n
# product of numbers from 1 to n
n * factorial(n-1)
'''


def fibonacci(n):
    '''
    Returns the n-th term in the following sequence
    0, 1, 1, 2, 3, 5, 8, 13, 21, ...
    The 1-th term is 0.

    Parameters:
    n: we need to find the n-th term in the sequence

    Return:
    The n-th term in the sequence
    '''
    if n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
    
'''
0 1 1 2 3 5 8 13 21 34 ...
fibonacci(n) # return the n-th term of the Fibonacci series
return (n-1)-th term + (n-2)-th term
'''

