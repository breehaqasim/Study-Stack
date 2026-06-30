def max2(a,b):
    if a>=b:
        return a
    return b

def max3(a,b,c):
    x = max2(a,b)
    return max2(x,c)

def max6(a,b,c,d,e,f):
    x = max3(a,b,c)
    y = max3(d,e,f)
    
    return max2(x,y)


print('Max of 6 is:', max6(12,15,10,31,2,8))