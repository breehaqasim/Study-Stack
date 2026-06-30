'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 12, Fall 2022

Some practice problems from Week 11.
'''

import math

def prime_factors(num: int) -> [int]:
    '''Returns the prime factors of num.

    https://www.hackerrank.com/test/co557stmbr5/questions/eet48o7hj84
    '''
    assert isinstance(num, int) and num >= 0
    # Identify the prime numbers upto num. These are candidate factors.
    primes = []
    for n in range(2, num + 1):
        # n is prime if it is not a multiple of any previous prime.
        is_prime = True
        for p in primes:
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    # Collect the primes that divide num.
    factors = []
    for p in primes:
        while num % p == 0:
            factors.append(p)
            num //= p
    return factors
    
    
def special_numbers(num: [int]) -> bool:
    '''
    A special number is equal to the sum of its prime factors.

    https://www.hackerrank.com/test/co557stmbr5/questions/eet48o7hj84

    0 - prime factors: [] . special
    1 - prime factors: [] . not special
    2 - prime factors: [2] . special
    3 - prime factors: [3] . special
    4 - prime factors: [2,2] . special
    5 - prime factors: [5] . special
    6 - prime factors: [2,3] . not special
    ...

    Return:
    list of all special numbers from 0 to num inclusive.
    '''
    assert isinstance(num, int) and num >= 0
    specials = []
    for i in range(0, num+1):
        if sum(prime_factors(i)) == i:
            specials.append(i)
    return specials



def is_neighbor(i: str, j: str) -> bool:
    '''Returns True if index j is a neighbor index i.

    neighbors of 0: 1, 3, 4
    neighbors of 1: 0, 2, 3, 4, 5
    neighbors of 2: 1, 4, 5
    neighbors of 3: 0, 1, 4, 6, 7
    neighbors of 4: 0, 1, 2, 3, 5, 6, 7, 8
    neighbors of 5: 1, 2, 4, 7, 8
    neighbors of 6: 3, 4, 7
    neighbors of 7: 3, 4, 5, 6, 8
    neighbors of 8: 4, 5, 7

    Helper function for the entry_time() function below.
    '''
    assert i in '012345678' and j in '012345678'
    if i == '0':
        neighbors = '134'
    elif i == '1':
        neighbors = '02345'
    elif i == '2':
        neighbors = '145'
    elif i == '3':
        neighbors = '01467'
    elif i == '4':
        neighbors = '01235678'
    elif i == '5':
        neighbors = '12478'
    elif i == '6':
        neighbors = '347'
    elif i == '7':
        neighbors = '34568'
    else:  # i == '8'
        neighbors = '457'
    return j in neighbors

def entry_time(s: str, keypad: str) -> int:
    '''Returns the time taken to enter the number represented by s in keypad.

    https://www.hackerrank.com/test/co557stmbr5/questions/acfp1s42qnc
    '''
    # Guards.
    for i in range(1, 10):
        assert str(i) in keypad
    for n in s:
        assert n in keypad
    # Identify the index of the first number. It takes 0 time to enter it.
    idx = keypad.index(s[0])
    time = 0
    # Iterate over the remaining numbers, and add their time accordingly:
    # - add 0 if the next number is the same as the current one
    # - add 1 if the next number is a neighbor of the current one
    # - add 2 otherwise
    for n in s[1:]:
        # Identify the index and add time.
        next_idx = keypad.index(n)
        if is_neighbor(str(idx), str(next_idx)):
            time += 1
        elif next_idx != idx:
            time += 2
        # Update the index for the next iteration.
        idx = next_idx
    return time
