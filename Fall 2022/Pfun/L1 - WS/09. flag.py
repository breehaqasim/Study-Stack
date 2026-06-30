'''
Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 11, Fall 2022

Problem solving.

Solves the flag problem from Exam 1.
'''

def flag(m, n):
    # Flag dimensions.
    w = max(m, n)
    h = min(m, n)
    line_width = 2*w - 1
    # Four types of lines in the flag.
    full_line = '* ' * (w-1) + '*'
    border_line = '*' + ' ' * (line_width - 2) + '*\n'
    num_spaces = (line_width - 3) // 2
    mid_star_line = '*' + ' ' * num_spaces + '*' + ' ' * num_spaces + '*\n'
    num_spaces = (line_width - (h-4) - (h-5) - 2) // 2
    center_line = '*' + ' ' * num_spaces + '* ' * (h-5) + '*' + ' ' * num_spaces + '*\n'
    # The lines containing the cross.
    flag_cross = mid_star_line * ((h-5)//2) + center_line + mid_star_line * ((h-5)//2)
    # The full flag.
    flag = full_line + '\n' + border_line + flag_cross + border_line + full_line
    return flag
