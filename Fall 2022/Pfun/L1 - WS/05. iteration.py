'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 8, Fall 2022
Iterative functions.
'''




# Mid 1 meetings
# HW 1 scores


# Iterable:
# - range(), for-loop 
# - string 


# Loops: for, while

def sum_first_n(n: int):
    '''
    Returns the sum of the first n numbers.

    Parameters:
    n: we need to sum the numbers from 1 to n

    Guard:
    n: must be positive

    Return:
    Sum of numbers from 1 to n inclusive.
    '''
    # Guard
    assert isinstance(n, int) and n > 0
    # Compute total.
    total = 0
    for i in range(n+1):
        total += i  # total = total + i
    return total


def count(n:int):
    '''
    Prints the numbers from 1 to n on separate lines.

    Parameters:
    n: we need to print the numbers from 1 to n

    Guard:
    n: must be positive

    Return:
    None
    '''
    # Guard
    assert isinstance(n, int) and n > 0
    # Print numbers.
    for i in range(1, n+1):
        print(i)

def factorial(n: int):
    '''
    Returns n!

    Parameters:
    n: we need to compute n!

    1 * 2 * 3 * 4 * ... * n

    Guard:
    n: must be positive

    Return:
    n!
    '''
    # Guard
    assert isinstance(n, int) and n > 0
    # Compute factorial.
    product = 1
    for i in range(1, n+1):
        product *= i
    return product

def fibonacci(n: int):
    '''
    Returns the n-th term in the following sequence
    0, 1, 1, 2, 3, 5, 8, 13, 21, ...
    The 1-th term is 0.

    Parameters:
    n: we need to find the n-th term in the sequence

    Guard:
    n: must be positive

    Return:
    The n-th term in the sequence
    '''
    # Guard
    assert isinstance(n, int) and n > 0
    # Compute n-th term.
    a = 0
    b = 1
    if n == 1:
        return a
    elif n == 2:
        return b
    for _ in range(n-2):
        # a, b = b, a + b
        c = a + b
        a = b
        b = c
    return b
