def hulk(n):
    '''
    Initiate all entries to "I hate that". Then make the following changes:
    - change the intervening values to "I love that".
    - replace last "that" to "it"
    '''
    lst = ["I hate that"] * n
    lst[1::2] = ["I love that"] * (n//2)
    lst[-1] = lst[-1].replace('that', 'it')
    return ' '.join(lst)
        
        
def fence(a, h):
    '''
    Aggregation: 1 or 2 for each height depending on how it compares to h
    '''
    return sum(2 if i > h else 1 for i in a)
    
def joke(guest, host, pile):
    '''
    Are the same letters included?
    '''
    return 'YES' if sorted(pile) == sorted(guest + host) else 'NO'

def snake(n, m):
    '''
    Initiate two types of lines: body and bend.
    Initiate all lines to the body type.
    Change relevant lines to the bend type.
    '''
    body = '#' * m
    bend = '.' * (m-1) + '#'
    snake = [body] * n
    snake[1::4] = [bend] * len(snake[1::4])
    snake[3::4] = [bend[::-1]] * len(snake[3::4])
    return '\n'.join(snake)


def general(a):
    '''

    - Need to get fist occurrence of max to index 0, and last occurrence of min
    to index, n-1.
    - Find index of max using list.find(). That is also the number of swaps.
    - Find index of min using list.find() in reversed list. That is also the
    number of swaps.
    - Total number of swaps is the sum of the swaps. Subtract 1 if max and min
    need to cross each other.
    '''
    max_pos = a.index(max(a))
    rev = a[::-1]
    min_pos = rev.index(min(rev))
    swaps = max_pos + min_pos
    if max_pos + min_pos >= len(a):
        swaps -= 1
    return swaps

