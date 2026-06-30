'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 7, Fall 2022

Recursive patterns in python turtle. Inspired by:
- https://cs.wellesley.edu/~fturbak/pubs/jcsc99.pdf
- https://cs111.wellesley.edu/~cs111/archive/cs111_spring15/public_html/notes/lectures/10_turtle_recursion_4up.pdf

python turtle documentation is available at:
https://docs.python.org/3/library/turtle.html
'''

def square(length):
    '''
    Draws a square with each side having the provided length.

    Uses a recursive helper function.

    Parameters:
    - length: length of each side of the square

    Return:
    None
    '''
    square_helper(length, 4)
    # repeat num_sides times:
    # - forward(length)
    # - turn(angle)

def square_helper(length, count):
    '''
    Recursively draws a square with the provided length.

    Parameters:
    - length: length of each side of the square
    - count: the number of sides remaining to be drawn

    Return:
    None
    '''
    if count == 0:
        return
    turtle.forward(length)
    turtle.left(90)
    square_helper(length, count-1)

def square1(length=100, count=4):
    '''
    Draws a square with each side having the provided length.

    Uses default arguments to eliminate the need of a helper function.

    Parameters:
    - length: length of each side of the square
    - count: the number of sides remaining to be drawn

    Return:
    None
    '''
    if count == 0:
        return
    turtle.forward(length)
    turtle.left(90)
    square1(length, count-1)

def polygon(num_sides, length=100):
    '''
    Draws a polygon with the provided num_sides and side length.

    Uses a recursive helper function.

    Parameters:
    - num_sides: number of sides of the polygon
    - length: length of each side of the polygon

    Return:
    None
    '''
    angle = 360 / num_sides
    polygon_helper(length, angle, num_sides)

def polygon_helper(length, angle, count):
    '''
    Recursively draws a polygon with the provided side length and exterior
    angle.

    Parameters:
    - length: length of each side of the polygon
    - angle: the exterior angle at each corner of the polygon
    - count: the number of sides remaining to be drawn

    Return:
    None

    '''
    if count == 0:
        return
    turtle.forward(length)
    turtle.left(angle)
    polygon_helper(length, angle, count-1)

    # Alternately
    # if count > 0:
    #     turtle.forward(length)
    #     turtle.left(angle)
    #     polygon_helper(length, angle, count-1)

def polygon_flower(num_petals, petal_sides, petal_length=100):
    '''
    Draws a "polygon flower" with the provided num_petals where each petal has
    the provided number of petal_length and petal_length.

    Uses a recursive helper function.

    Parameters:
    - num_petals: the number of petals in the flower
    - petal_sides: the number of sides in each petal
    - petal_length: the length of each side of a petal

    Return:
    None

    '''
    angle = 360 / num_petals
    polygon_flower_helper(petal_length, petal_sides, angle, num_petals)

def polygon_flower_helper(length, sides, angle, count):
    '''
    Recursively draws a "polygon flower" with the provided side length, number
    of sides, exterior angle, and num_petals.


    Parameters:
    - length: length of each side of a petal
    - sides: the number of sides on each petal
    - angle: the angle between successive petals
    - count: the number of petals remaining to be drawn

    Return:
    None

    '''
    if count == 0:
        return
    polygon(sides, length)
    turtle.left(angle)
    polygon_flower_helper(length, sides, angle, count-1)
    
    # compute angle
    # repeat num_petals times:
    # - draw polygon(petal_length, petal_sides)
    # - turn angle
