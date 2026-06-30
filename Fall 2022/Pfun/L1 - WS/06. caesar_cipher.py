'''Author: Waqar Saleem
Email: waqar.saleem@sse.habib.edu.pk
Date: Week 9, Fall 2022

Caesar cipher
- illustrates: problem solving, iteration over a string, modular arithmetic,
conditional, use of ASCII code, unit testing.
'''

def encrypt_5(plaintext: str):
    '''
    Encrypts plaintext using substitution cipher with key equal to 5.

    plain:  ABCDEFGHIJKLMNOPQRSTUVWXYZ
    cipher: FGHIJKLMNOPQRSTUVWXYZABCDE

    Parameters:
    plaintext: the text to encrypt
    
    Returns:
    encryption of plaintext
    '''
    ciphertext = ""  # contains the ciphertext
    plaintext = plaintext.upper()
    for p in plaintext:
        c = p  # p encrypts to itself by default
        ascii_code = ord(p)
        if 65 <= ascii_code <= 90: # p is a letter, perform encryption
            ascii_code += 5
            # # Wrap-around if needed.
            # if ascii_code > 90:
            #     ascii_code -= 26
            # Use modulo for wrap-around.
            ascii_code = ((ascii_code - 65) % 26) + 65
            c = chr(ascii_code)
        # Append encrypted letter to cipher text.
        ciphertext += c
    return ciphertext
    
# how to check if a string is a letter? - ASCII of letters are between 65 and 90.
# how to check for case? - convert string to upper case
# how to shift a letter by 5? add 5 to ASCII, subtract 26 if ASCII > 90

def test_encrypt_5():
    '''
    Tests the encrypt_5 function.
    '''
    # Check that numbers encrypt to themselves.
    for i in range(1000):
        assert encrypt_5(str(i)) == str(i)
    # Check that special characters encrypt to themselves.
    for c in "~`!@#$%^&*()-_+=<>,.;:'{}[]\|?/ ":
        assert encrypt_5(c) == c
    # Check encryption of some strings in mixed case and containing special
    # characters.
    assert encrypt_5("A") == "F"
    assert encrypt_5("Hello World!") == "MJQQT BTWQI!"
    assert encrypt_5("In Yohsin Hall at 9.") == "NS DTMXNS MFQQ FY 9."
    assert encrypt_5("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == "FGHIJKLMNOPQRSTUVWXYZABCDE"
    # Print message if all asserts passed.
    print('All tests passed!')
