#Function definition
def rotate(str1, r, op):
 i=0 # indexing 

 if op == 1:
    while i < len(str1):
      print (chr(ord(str1[i])-r), end= '')
      i +=1
    print ("")

 elif op == 2:
    while i < len(str1):
      print (chr(ord(str1[i])+r), end= '')
      i +=1
    print ("")

 elif op == 3:
    i = -1
    while i >= -(len(str1): 
      print(str1[i], end= '')
      i -=1
    print ("")

#User input    
str1= input('Enter string: ')
op =  int(input('Enter operation:1(negative), 2(positive) 3(backwards):'))
if op != 3:
  r = int(input('Enter rotation: '))
else: 
  r = 0

#Function call  
rotate(str1,r, op)
  