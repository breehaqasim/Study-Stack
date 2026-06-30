# -------------------------------------------------------
# Question 2 - Pattern 3
# -------------------------------------------------------
def pattern(n):
    num = 1
    for line_no in range(1,n+1):
        for j in range(line_no+(line_no-1)):
            print(num, end=" ")
            num += 1
        print()
# -------------------------------------------------------
# Question 3 - Slow Conceal 
# -------------------------------------------------------
def slow_conceal_1(s):
    #using for loop
    for i in range(len(s)):
        end_index = len(s)-i
        print(s[0:end_index])

def slow_conceal_2(s):
    #using while loop
    end_index = len(s)
    while end_index > 0:
        print(s[:x])
        end_index = end_index - 1

# -------------------------------------------------------
# Question 7 - Recursively Devowelify
# -------------------------------------------------------
def devowelify(s):
    vowels = "aeiouAEIOU"
    if len(s) == 0:
        return ""
    else:
        if s[0] in vowels:
            char = ""
        else:
            char = s[0]
        return char + devowelify(s[1:])

def devowelify_loops(s):
    #same problem but using loops
    vowels = "aeiouAEIOU"
    output = ""
    for i in s:
        if i not in vowels:
            output = output + i
    return output

# -------------------------------------------------------
# Question 8 - Recursively find length of a string
# -------------------------------------------------------
def length(s):
    if s == "":
        return 0
    else:
        return 1 + length(s[1:])