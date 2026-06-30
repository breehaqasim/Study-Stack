'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 11, Fall 2022

Statistics on a list of numbers: mean, median, and mode.
'''

def mean(lst):
    '''Returns the mean of the elements in lst.
    '''
    assert len(lst) > 0
    # total = 0
    # for i in lst:
    #     total += i
    # return total / len(lst)
    return sum(lst) / len(lst)

'''
1:  _
i:  0

3:  _ , _, _
i:  1

5:  _, _, _, _, _
i:  2

7:  _, _, _, _, _, _, _
i:  3
i = n // 2

0:  
i:  Error

2:  _ , _
i,j:  0,1

4:  _, _, _, _
i,j:  1,2

6:  _, _, _, _, _, _
i,j:  2,3

8:  _, _, _, _, _, _, _, _
i,j:  3,4

10:  _, _, _, _, _, _, _, _, _, _
i,j:   4,5

'''


def median(lst):
    '''Returns the median element in lst.
    '''
    assert len(lst) > 0
    lst = sorted(lst)
    n = len(lst)
    i = n // 2
    if n % 2 == 1:
        # lst is of odd length
        median = lst[i]
    else:
        # lst is of even length
        j = i - 1
        median = (lst[i] + lst[j]) / 2
        # median = mean(lst[i:i+2])
    return median

def mode(lst):
    '''Returns the mode of lst.
    '''
    assert len(lst) > 0
    mode = lst[0]
    mode_count = 1
    for i in lst:
        count = lst.count(i)
        if count > mode_count:
            mode_count = count
            mode = i
    return mode
