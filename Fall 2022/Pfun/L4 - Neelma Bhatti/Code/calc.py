
# Adding two numbers
def add(x,y):
    return num1 + num2

# Subtracting two numbers
def subtract(x, y):
    return x - y

# Multiplying two numbers
def multiply(x, y):
    return x * y

# Dividing two numbers
def divide(x, y):
    return x / y


# take input from the user
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

choice = input("Select operation: + , - , * , /: ")

#Compute two numbers according to the operation selected
if choice == '+':
    print(num1, "+", num2, "=", add(num1, num2))

elif choice == '-':
    print(num1, "-", num2, "=", subtract(num1, num2))

elif choice == '*':
    print(num1, "*", num2, "=", multiply(num1, num2))

elif choice == '/':
    print(num1, "/", num2, "=", divide(num1, num2))
        

else:
    print("Invalid Input")
