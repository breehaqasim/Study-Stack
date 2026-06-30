#include <iostream>

/* while loop
    - It looks simplified, but it has all the information it needs to run.
    - Note the syntax.
*/

int main () 
{
    int user_choice = -1;
    while (user_choice != 0)
    {
        std::cout << "Please enter a number = ";
        std::cin >> user_choice;
        
        std::cout << "user_choice = " << user_choice << std::endl;
    }

    return 0; 
}