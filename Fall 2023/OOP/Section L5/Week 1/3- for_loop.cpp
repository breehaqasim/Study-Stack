#include<iostream>

/* for loops
    
    for ( initialization_expression ; test_expression ; increment_expression ) 
    {
        // loop body
    }
    
    Even though it is called an "increment expression", it can have other arithmetic operations as well. Please check.
    
    Note that those three expressions inside the parentheses of the for loop are optional. 
*/

int main () 
{
    // =============================================================
    // LOOP VARIABLE DECLARED INSIDE THE FOR LOOP.
    // =============================================================
    
    // Define the loop variable inside the loop declaration. 
    // Generally better to do this.
    
    for (int counter2{0}; counter2 > -15; counter2 -= 3) 
    {
        // int new_variable{-214};
        std::cout << counter2 << '\n';
    }

    // std::cout << "This is the last value of counter2 = " << counter2 << std::endl;
    // std::cout << "new_variable = " << new_variable << std::endl;
    

    /* NOTE:
       1- Cannot access "counter2" and "new_variable" outside the loop.
    */

    /* NOTE:
        1- all of the expressions in the parentheses are optional.
    */


    // =============================================================
    // LOOP VARIABLE DECLARED OUTSIDE THE FOR LOOP.
    // =============================================================

    // for counter in range(1, 5):      in Python
    
    // int counter; 
    // for (counter = 0; counter < 15; counter++) {
    //     std::cout << counter * counter << '\t';
    // }
    // std::cout << std::endl << "This is the last value of counter = " << counter << std::endl;
    
    /* NOTE: 
       1- "counter" does go to 15.
       2- "counter" can be accessed outside the loop, because it was initialized outside the for loop.
       3- Single statement loop bodies don't need braces.
    */
    
    return 0;
}

