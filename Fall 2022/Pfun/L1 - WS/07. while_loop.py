'''Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 9, Fall 2022
'''

def while_example_1():
    total = 0
    entered_number = 1
    while entered_number > 0:
        entered_number = int(input('Please enter a positive integer, 0 or negative to exit: '))
        if entered_number > 0:
            total = total + entered_number
    return total

def while_example_2():
    total = 0
    while True:
        entered_number = int(input('Please enter a positive integer, 0 or negative to exit: '))
        if entered_number > 0:
            total = total + entered_number
        else:
            break
    return total

def for_example_1(n: int):
    ''' Returns 1 + 2 + 3 + ... + n
    '''
    # Guard
    assert isinstance(n, int)
    # Iterate n times - for and while versions
    total = 0
    # for i in range(1, n+1):
    #     total += i
    i = 1
    while i < n+1:
        total += i
        i += 1
    return total
