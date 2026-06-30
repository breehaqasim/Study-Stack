def race(f1up, f1down, f2up, f2down):
    f1, f2 = 0, 0
    positions = [(0,0)]
    done = False
    winner = ' '
    while not done:
        f1 += f1up
        f2 += f2up
        jumps = len(positions)
        if f1 >= 1000 and f2 >= 1000:
             positions.append(('OUT', 'OUT'))
             winner = 'Tie in ' + str(jumps)+ ' turns!'
             done= True
        elif f1>= 1000:
             f2 -= f2down
             positions.append(('OUT', f2))
             winner = 'Frog Prime wins in '+str(jumps)+ ' turns!'
             done = True
        elif f2 >= 1000:
            f1 -= f1down
            positions.append((f1,'OUT'))
            winner = 'Frogatron wins in '+str(jumps)+ ' turns!'
            done = True
        else:
            f1 -= f1down
            f2 -= f2down
            positions.append((f1,f2))
    print(positions)
    print(winner)

race(50,2,1000,4)

        
#race(f1up,f1down,f2up,f2down)
