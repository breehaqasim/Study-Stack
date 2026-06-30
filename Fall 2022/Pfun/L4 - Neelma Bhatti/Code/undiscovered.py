def word_search(word, sentence):
    word = word.lower()
    sentence = sentence.lower()
    # Book-keeping information.
    n = len(word)
    j = 0 
    for i in range(len(sentence)):
        # Ignore non-letters.
        if not sentence[i].isalpha():
            continue
        # A match is found for the current index of word. 
        if sentence[i] == word[j]:
            if j == 0:
                start = i
            j += 1
            if j == n:
                return [start, i+1]
        else:
            # Reset matching to the start of word.
            j = 0
    return []

print(word_search('Japan', 'aaj apa narangi laayi hain'))