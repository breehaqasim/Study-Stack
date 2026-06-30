def biggest_bucket(corpus):
    buckets = {}
    mode = []
    mode_freq = 0
    for word in corpus.lower().split():
        for letter in word:
            if letter.isalpha():
                buckets[letter] = buckets.get(letter, 0) + 1
                if buckets[letter] > mode_freq:
                    mode_freq = buckets[letter]
                    mode = [[letter, buckets[letter]]]
                elif buckets[letter] == mode_freq:
                    mode.append([letter, buckets[letter]])
                break
    return sorted(mode)

print(biggest_bucket('A noisy noise annoys only a **nosy** oyster,'))