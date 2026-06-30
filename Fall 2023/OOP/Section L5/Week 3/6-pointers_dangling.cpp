#include<iostream>

int main()
{
    /* DANGLING POINTERS

        - It is a pointer that is holding the address of an object that is no longer valid.
            e.g. because it has been destroyed.

                - For example, a variable that has gone out of scope.

                - A variable that was local to a function.

                - An entity whose memory has been dynamically deallocated.

        
        - Dereferencing a dangling pointer will lead to UNDEFINED behavior.

            - You don't know what could happen.

            - GOOD PRACTICE: Set such a pointer to nullptr.
    */

    int x{ 5 };
    int* ptr{ &x };

    std::cout << *ptr << '\n'; // valid

    // Pausing...
    std::cin.get();

    // A code block.
    {
        /* QUESTION
            - Will 'y' be available after the code block?
            - Will 'ptr' be available after the code block?
        */

        int y{ 6 };
        ptr = &y;

        std::cout << *ptr << '\n'; // valid

        // Pausing...
        std::cin.get();
    } 




    // y goes out of scope, and ptr is now dangling
    std::cout << *ptr << '\n'; // undefined behavior from dereferencing a dangling pointer

    /* NOTE
        
        - The last print above may output 6 (value of y).

        - But it could also not, as the entity that "ptr" was pointing at went out of scope. 
            - "y" was destroyed at the end of the inner block, leaving "ptr" dangling.
    */

    return 0;
}