#include<iostream>
using std::cout; using std::cin; using std::endl;


// Associated Reading: https://www.learncpp.com/cpp-tutorial/function-template-instantiation/ 
    // Section: "Instantiated functions may not always compile"


template <typename T>
T addOne(T x)
{
    return x+1;
}

int main()
{
    std::string hello { "Hello, world!" };
    cout << addOne<std::string>(hello) << endl;

    
    /* DISCUSSION
        - A template function will be instantiated, where template type is std::string.
        - However, adding an integer to a std::string doesn't make sense, so an error comes up.
        - In this case, avoid calling the template function with a std::string.

        - Note that the error is not due to how templates work.
            - It is due to the fact that an operator+ overload doesn't exist that could add a std::string and an int.
    */

    return 0;
}

/* NOTE - GENERIC PROGRAMMING
    - Because template types can be replaced with any actual type, template types are sometimes called GENERIC TYPES. 
    
    - And because templates can be written agnostically of specific types, programming with templates is sometimes called GENERIC PROGRAMMING. 
    
    - Whereas C++ typically has a strong focus on types and type checking, in contrast, ...
      ... generic programming lets us focus on the logic of algorithms and design of data structures without having to worry so much about type information.
*/

// ====================== OPTIONAL SECTION ======================

/* NOTE - MULTIPLE FILES
    - When working with multiple files, place your template definitions in header files.
        This is different to the normal practice of keeping the function defs in a separate .cpp file, and their declarations in a corresponding .h file.

    - So that when you include such a header file, ...
      ... the file in which you included it would then have this template definition, and can make use of it.
*/
