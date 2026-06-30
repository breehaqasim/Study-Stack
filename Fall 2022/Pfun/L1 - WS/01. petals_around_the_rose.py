# Author: Waqar Saleem
# Email: waqar.saleem@sse.habib.edu.pk
# Date: Week 4, Fall 2022
#
# Solution(s) to the "Petals around the Rose" game
# https://illuminations.nctm.org/lessons/petals/petals.htm
#
# Respect the potentate of the rose. Do not disseminate the solution. Let others
# experience the pleasure of figuring it out on their own!


def petals_around_the_rose(a, b, c, d, e):
    '''
    Returns the number of petals on the dice.

    Parameters:
    - a: the number on the first die
    - b: the number on the second die
    - c: the number on the third die
    - d: the number on the fourth die
    - e: the number on the fifth die

    Return:
    the total number of petals on all 5 dice
    '''
    petals = count_petals(a) + count_petals(b) + count_petals(c) + count_petals(d) + count_petals(e)
    return petals

def count_petals(num):
    '''
    Returns the number of petals corresponding to a number on a die.

    Parameters:
    - num: the number on the die

    Return:
    the number of petals corresponding to num.
    '''
    petals = 0
    if num % 2 == 1: # 1, 3, 5
        petals = num - 1
    return petals

print(petals_around_the_rose(dice1,2,3,4,5))

