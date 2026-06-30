#single argument
def enrollment (name):
  print (name,' is enrolled in Pfun')
   
#enrollment('abc')

#multiple arguments
def enrollment (name, course):
  print (name,' is enrolled in ', course)

enrollment('abc','Pfun')

#name = input ("Enter your name: ")
#course = input ("Course you are enrolled in: ")
#enrollment(name, course)

#default arguments
def enrollment ( name, course = 'RhetCom'):
  print (name,' is enrolled in ', course)
   
#enrollment('abc', course= 'Pfun')


#positional vs keyword arguments
def enrollment ( name, id = '453', course = 'RhetCom'):
  print (name, 'whose course ID is ', id, ' is enrolled in ', course)
  
#enrollment('abc', course= 'Pfun')


#arbitary arguments
def enrollment (*name):
  print ("Student(s) enrolled in PFun: ", name)

name = ('a', 'b')
enrollment(name)
