'''
CS 101 Fall 2022 Exam 2
Saturday, 12 November, 2022

Solutions for Exam A and Exam B.
'''

############
## Exam A ##
############

# Not_bad.
def not_bad(s):
    '''Approach: check if 'not' is followed by 'bad' in s. If so, replace the
    appropriate slice in s with 'good'.
    '''
    # Find 'not' in s and store the starting index. Failure if 'not' is not in s
    # - return s as is.
    start_idx = min(s.find('not'), s.find('Not'))
    if start_idx == -1:
        return s
    # Find 'bad' in s after the occurrence of 'not' and store the starting
    # index. Failure if 'bad' is not found - return s as is.
    end_idx = min(s.find('bad', start_idx), s.find('Bad', start_idx))
    if end_idx == -1:
        return s
    # Success: construct new string as slice before 'not' + 'good' + slice after
    # bad.
    return s[:start_idx] + 'good' + s[end_idx+3:]


# Multiplicursion table.
def multiplication_table(n, k):
    '''Approach: recursively print the table for (n, k-1) and then print the line
    for n * k.
    '''
    if k > 1:
        multiplication_table(n, k-1)
    print(n, '*', k, '=', n*k)
    

# Edit distance.
def edit_distance(s1, s2):
    '''Approach: Identify the case. For case 1, count the number of mismatched
    indices. For case 2 if s1 is shorter, iteratively generate slices of s2 and
    recursively compute the distance of each with s1. For case 2 if s2 is
    shorter, recursively compute the distance with s1 and s2 swapped.
    '''
    # Normalize the case.
    s1 = s1.lower()
    s2 = s2.lower()
    # Determine the case.
    l1 = len(s1)
    l2 = len(s2)
    if l1 == l2:  # case 1: equal length
        # distance is the number of mismatched indices.
        distance = 0
        for i in range(l1):
            if s1[i] != s2[i]:
                distance += 1
    elif l1 < l2:  # case 2: unequal length, s1 is shorter
        # first compute the distance with the substrings and record the minimum.
        distance = l2  # initialize with a large value.
        for i in range(l2-l1+1):
            distance = min(distance, edit_distance(s1, s2[i:i+l1]))
        # add the difference in length.
        distance += l2 - l1
    else: # case 2: unequal length, s2 is shorter
        # call the function with the strings swapped.
        distance = edit_distance(s2, s1)
    return distance


# Exchange rate woes.    
def max_gain(lst):
    '''Approach: check all possible intervals and record the maximum.
    '''
    n = len(lst)
    largest = lst[0]  # initialize with the first d.
    for i in range(n):
        for j in range(i, n+1):
            largest = max(largest, sum(lst[i:j]))
    return largest
                        

# A family affair.
def selek_sees(num, word):
    '''Approach: the reading is valid if every digit in num is always replaced with
    the same letter in word.
    '''
    n = len(num)
    # Check for all digits.
    for digit in range(10):
        # Nothing to do if a digit does not appear in num.
        if digit not in num:
            continue
        # Find the correspodning letter for the first occurrence of digit.
        index = num.index(digit)
        letter = word[index]
        # Failure if digit corresponds with a different letter from here on.
        for i in range(index, n):
            if num[i] == digit and word[i] != letter:
                return False
    # No failure found - success.
    return True


############
## Exam B ##
############

# Theft of character.
def blanks(s):
    '''Approach: check each letter in s and construct the remaining string through
    slicing.
    '''
    for i in range(len(s)):
        print(s[i], ':', s[:i] + '_' + s[i+1:])

# Integer to text.
def int_to_text(n):
    '''
    Approach: Recursively convert n digit by digit, until a single digit remains.
    '''
    last_digit = n % 10
    n //= 10
    text = chr(last_digit + 48)
    if n > 0:
        text = int_to_text(rest_of_n) + text
    return text

# Edit distance - same as in Exam A above.

# Stock market fluctuations - same as the "Exchange rate woes" problem in Exam A
# above.

# The undiscovered country.
def word_search(word, sentence):
    '''Approach: Iterate over sentence and keep track of the portion of word that
    is matched so far.
    '''
    # Normalize the case of both strings.
    word = word.lower()
    sentence = sentence.lower()
    # Book-keeping information.
    n = len(word)
    j = 0  # the current index in word which has to be matched.
    # Iterate over sentence and keep track of the indices that form a match with
    # word.
    for i in range(len(sentence)):
        # Ignore non-letters.
        if not sentence[i].isalpha():
            continue
        # A match is found for the current index of word. 
        if sentence[i] == word[j]:
            # Note if this is the beginning of word. Ensure that that the next
            # match is checked for the next index in word. Success if all of
            # word has been matched.
            if j == 0:
                start = i
            j += 1
            if j == n:
                return [start, i+1]
        # A match is not found or an ongoing incomplete match is interrupted.
        else:
            # Reset matching to the start of word.
            j = 0
    # No match was found in sentence.
    return []
