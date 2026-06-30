# Author: Waqar Saleem
# Email: waqar.saleem@sse.habib.edu.pk
# Date: Week 5, Fall 2022
#
# Solution(s) to the "guess the number" game described at the start of
# https://www.risingstars-uk.com/blog/march-2018/blog-march-2018-magic-maths-blog


def guess_the_number(a, b, c, d):
    '''
    Returns the number corresponding to the containing boxes.

    A is the top-left box.
    B is the top-right box.
    C is the bottom-right box.
    D is the bottom-left box.

    Parameters:
    - a: True if the number is contained in A; False otherwise.
    - b: True if the number is contained in B; False otherwise.
    - c: True if the number is contained in C; False otherwise.
    - d: True if the number is contained in D; False otherwise.

    Return:
    the number corresponding to the containing boxes.
    '''
    num = 0
    if a == True:
        num = num + 8
    if b == True:
        num = num + 4
    if c == True:
        num = num + 1
    if d == True:
        num = num + 2
    return num
