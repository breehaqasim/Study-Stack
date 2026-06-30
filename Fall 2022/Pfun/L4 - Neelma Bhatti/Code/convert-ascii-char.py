#Converts ASCII code given by user to its equivalent character
def ASCII_to_char (a):
    ascii = a
    print(str(ascii) + " represents " + (chr(ascii)) + " in ASCII code")

#Converts the character given by user to its equivalent ASCII code
def char_to_ASCII (a):
    ch = a
    print ("The ASCII code of " + ch + " is " + str(ord(ch)))

#Ask user for the conversion they want to perform
print ("1: ASCII code to character")
print ("2: character to ASCII code")
op = input ("Enter the function: ")

#call appropriate function based on selection
if op == '1':
    a = int(input("Enter the ASCII code to convert it to character: "))
    ASCII_to_char(a)

elif op == '2':
    a = input("Enter the character whose ASCII code you want to know: ")
    char_to_ASCII(a)

else: 
    print ("Invalid selection")



