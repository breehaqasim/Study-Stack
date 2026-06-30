# Heroes amongst Zeroes
def rec_distance(grid):
    pos = []
    for row in range(n):
        for col in range(m):
            if grid[row][col] == '1':
                pos.append((row,col))
    return abs(pos[0][0] - pos[1][0]) + abs(pos[0][1] - pos[1][1])

# I love you 3000!
def weeks_from_days(days):
    if days < 7:
        return 0
    return 1 + weeks_from_days(days - 7)

def party_time(line):
    date = line.split('-')
    for i in range(3):
        date[i] = int(date[i])
    days = 30 - date[0] + 30 * (12 - date[1]) + 1
    weeks = weeks_from_days(days)
    year = date[2] + 1
    if days <= 1:
        print("Not enough time to plan the party, Tony! Try next year!")
    elif weeks < 1:
        print("There are only", days, "day(s) left to the New Year,", year)
    else:
        print("There are around", weeks, "more weeks to the New Year,", year)

# Yohsinator
def is_fragment(w, f):
    return f.lower() in (w*100).lower()

# Know your shopping cart!
def get_frequent_items(purchases, searchitem):
    # Collect items bought with searchitem
    items = []
    for purchase in purchases:
        for i in range(len(purchase)):
            if purchase[i] == searchitem:
                items += purchase[:i] + purchase[i+1:]
    # If no items were collected.
    if not items:
        return []
    # Count the quantity of each collected item.
    count = {}
    for item in items:
        count[item] = count.get(item, 0) + 1
    # Return items with the maximum quantity.
    max_quantity = max(count.values())
    items = []
    for item in count:
        if count[item] == max_quantity:
            items.append(item)
    return items

# CS Raj Namanzoor
def longest_portion(sequence):
    # Count the number of consecutive repetitions, or the length of "run"s.
    previous = None  # last seen item in the sequence - nothing seen yet.
    count = 0  # length of the current run - no run has started yet
    runs = []  # lengths of runs - no run encountered yet
    for a in sequence:
        # repeated character - increment run count
        if a == previous:
            count += 1
        # different character - previous run has ended. store it and set up new run.
        else:
            if count > 0:
                runs.append(count)
            count = 1
            previous = a
    # store the last run
    runs.append(count)
    # Look at consecutive pairs of runs. A valid half has length equal to the
    # smaller value in a pair. The length of the portion is twice as
    # much. Return the max such value.
    max_run = 0
    for i in range(len(runs)-1):
        l = min(runs[i], runs[i+1])
        max_run = max(l, max_run)
    return 2 * max_run

