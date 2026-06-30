'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 10, Fall 2022

Lists for math operations.
- vector addition
- martrix addition
'''

'''
_   _       _   _       _   _ 
| 2 |       | 0 |       | 2 | 
| 5 |       | 3 |       | 8 | 
| 1 |  +    | 6 |   =   | 7 | 
| 0 |       | 9 |       | 9 | 
| 7 |       | 4 |       | 11| 
-   -       -   -       -   - 
'''

def add_vectors(v1: [int], v2: [int]) -> [int]:
    '''Returns the sum of vectors, v1 and v2.
    '''
    # Check that dimensions match.
    d = len(v1)
    assert d == len(v2)
    # Compute the sum.
    # Approach: Initialize result vector with 0's and replace them.
    # result = [0] * d
    # for i in range(d):
    #     result[i] = v1[i] + v2[i]
    # Approach: Create blank result vector and populate it.
    result = []
    for i in range(d):
        result.append(v1[i] + v2[i])
    return result

'''
_         _       _              _        _              _ 
| 2 1 5 2 |       | 1  0  1  0   |        | 3  1  6  2   | 
| 5 4 4 5 |       | 3  1  3  3   |        | 8  5  7  8   | 
| 1 0 7 1 |  +    | 6  6  3  5   |   =    | 7  6 10  6   | 
| 0 3 0 0 |       | 9  3  7  2   |        | 9  6  7  2   | 
| 7 2 1 7 |       | 4  4  2  1   |        |11  6  3  8   | 
-         -       -              -        -              - 

[  [2, 1, 5, 2],[ 5, 4, 4, 5], [ 1, 0, 7, 1], [0, 3, 0, 0], [ 7, 2, 1, 7] ]

'''
def add_matrices(m1: [[int]], m2: [[int]]) -> [[int]]:
    '''Returns the sum of matrices, m1 and m2, stored in row-major form.
    '''
    # Check that dimensions match.
    num_rows = len(m1)
    assert len(m1) == len(m2)
    for i in range(num_rows):
        assert len(m1[i]) == len(m2[i])
    # Compute the sum.
    # Approach: Create empty result matrix and populate it.
    num_cols = len(m1[0])
    result = []
    for r in range(num_rows):
        # Populate row.
        row = []
        for c in range(num_cols):
            row.append(m1[r][c] + m2[r][c])
        # Add row to matrix.
        result.append(row)
    return result
