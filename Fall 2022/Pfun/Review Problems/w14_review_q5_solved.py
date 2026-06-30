# Week 14 Review Questions
# Q5 Solution
# Good luck for the exam!

def biggest_bucket(corpus):
    newstr = ''
    for i in corpus:
        if i.isalpha() or i == " ":
            newstr += i
            
    newstr = newstr.lower()
    newstr = newstr.split() #list of words in the corpus
    
    freq = {}
    for word in newstr:
        char = word[0]
        if char in freq:
            freq[char] = freq[char]+1
        else:
            freq[char] = 1
    
    max_chrs = []
    keys = list(freq.keys()) #chars
    values = list(freq.values()) #count
    
    max_val = values[0]
    max_ind = [0] # [0,5,15] 
    
    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_ind = [i]
        elif values[i] == max_val and i not in max_ind:
            max_ind.append(i)
    
    result = []
    for index in max_ind:
        temp = []
        temp.append(keys[index])
        temp.append(values[index])
        result.append(temp)
    result.sort()
    return result
