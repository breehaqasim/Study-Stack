x = 20  #global x
y = 10  #global y

#local and global variables
def scope_demo():
	x = 10 # local variable x
	print("Value inside function:",x)

def scope_demo2():
	y = 20 # local variable y
	print("Value inside function:", y)

scope_demo()
scope_demo2()
print("Value outside function:",x, y)

#print (x)
