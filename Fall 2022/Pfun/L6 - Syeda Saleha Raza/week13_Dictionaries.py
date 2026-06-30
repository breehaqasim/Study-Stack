# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:32:05 2022

@author: saleha.raza
"""

# Counting letters
# anagrams
# most occuring letters


def mode(str1):
    d = {}
    for c in str1.lower():
        if c.isalpha():        
            d[c]= d.get(c,0) + 1
    
    print(d)
    
    m = 0
    for k in d:
        if d[k] > m:
            m = d[k]
            
    lst =[]

    for k in d:
        if d[k] == m:
            lst.append(k)
            
    return lst


print(mode('this is again a sample text'))    
    
    

# def anagram(s1,s2):
#     d1 = {}
#     d2 = {}

#     for c in s1.lower():
#         if c.isalpha():        
#             d1[c]= d1.get(c,0) + 1

#     for c in s2.lower():
#         if c.isalpha():        
#             d2[c]= d2.get(c,0) + 1
            
#     if len(d1) != len(d2):
#         return False

#     for k in d1:
#         if d1[k] != d2.get(k,-1):
#             return False
#     return True
            

# print(anagram('debit  card','bad  credit'))



def countLetters(s):
    s = s.lower()
    letterCount = dict()
    for l in s:
        if l in letterCount:
            letterCount[l] += 1
        else:
            letterCount[l] = 1

    return letterCount


print(countLetters('I hate programming but I love python'))


wordDict = {'happy':'(*•‿•*)', 'sad':'(⁍﹏⁌)','silent':':-|','confused':'◔_◔','angry':'٩ (╬ʘ益ʘ╬) ۶','crying':'(-̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥᷄_-̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥̥᷅ )','hungry':'♨(⋆‿⋆)♨'}

comments = 'I am sad. I feel like crying. The exam made me confused. It went so bad. But Ms. Saleha is soo nice. She will make us happy.'
comments2=''

sentences = comments.split('.')
for s in sentences:
    comments2 +=s
    words = s.split(' ')
    for i in words:
        comments2 += wordDict.get(i,'')
    comments2 +='.'    
print(comments2)





    
    