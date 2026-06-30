# Week 5 - Review and Lab Exercises for Practice
# Question 1 - Solution

def is_input_correct(field):
    if field.isdigit():
        if int(field) >= 0 and int(field) <= 999:
            return True
        return False
    elif len(field) == 1 and ord(field) >= ord("A") and ord(field) <= ord("Z"):
        return True
    return False

def next_element(item):
    if item.isdigit():
        new_elem = int(item)
        if new_elem >= 0 and new_elem < 999:
            new_elem += 1
        else: 
            new_elem = 0
        new_elem = str(new_elem)
        if len(new_elem) < 3:
            new_elem = new_elem.zfill(3)
        return new_elem
    else:
        new_elem2 = item
        # new_elem2 = chr(ord(new_elem2) + 1)
        if new_elem2 == 'Z':
            new_elem2 = 'A'
            return new_elem2
        return (chr(ord(new_elem2)+1))
    
def main():
    a = input()
    bcd = input()
    e = input()
    
    next_a = a
    next_bcd = bcd
    next_e = e
    
    if is_input_correct(a) and is_input_correct(bcd) and is_input_correct(e):
        next_bcd = next_element(bcd)
        if next_bcd == '000':
            next_e = next_element(e)
            if next_e == 'A':
                next_a = next_element(a)
        if next_a == 'A' and next_bcd == '000' and next_e == 'A':
            print("The next license plate is: Not possible")
            return 
        print("The next license plate is: "+next_a+"-"+next_bcd+"-"+next_e)
    else:
        print("Invalid Entry")
        return 
        
        
            
        
             
    

if __name__ == "__main__":
    main()