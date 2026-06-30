#include<iostream>
using std::cout; 
using std::cin; 
using std::endl;

// Associated Reading: https://www.learncpp.com/cpp-tutorial/function-template-instantiation/

/* DISCUSSION
    - Function templates are not actually functions -- their code isn’t compiled or executed directly. 
    
    - Instead, function templates have one job: to generate functions (that are compiled and executed).
*/

// Note that each of the two functions below have their own template parameter declaration.

template <typename T>
T max(T x, T y)
{
    // return (x > y) ? x : y;

    if(x > y) {
        return x;
    }
    else {
        return y;
    }
}

template <typename U>
bool is_left_max(U x, U y)      // note that you can mix and match "known" types with template params.
{
    return x > y;
}


int main()
{
    int x { 6 };
    int y { 8 };

    // max<actual_type>(arg1, arg2);    // actual_type is some actual type, like int or double
       max<int>        (x   , y)   ;

    /* DISCUSSION
        - "actual_type" above is called a TEMPLATE ARGUMENT.
            - In the example above, it is "int".
            - Template argument specifies the actual type that will be used in place of template type T.

        - When the above statement is encountered, ...
            ... a new function max<int>(int, int) will be created, BECAUSE one doesn't already exist.

            - This implies that if one had already existed, another won't be created.

        - This process of function creation from function template definitions (by specifying a type) ...
            ... is called FUNCTION TEMPLATE INSTANTIATION (or INSTANTIATION).

        - This is an e.g. of an IMPLICIT INSTANTIATION (since this function was created from a function call).

        - The function created is called a FUNCTION INSTANCE (or TEMPLATE FUNCTION).
            - These functions are normal functions in all regards.
    */

    return 0;
}

// ==========================================================================

// ====================== OPTIONAL SECTION STARTS ======================

/* NOTE
    - Function instantiation is straightforward
        - Compiler clones the template function definition, ... 
          ... and replaces the template type with the specified type.

    - So you could think that when the compiler sees, max<int>(1, 2), ...
      ... the instantiated function looks like below.
*/

/*
// a declaration for our function template (we don't need the definition any more)
template <typename T>
T max(T x, T y);


template <>
int max<int>(int x, int y)      // the generated function max<int>(int, int)
{
    return (x > y) ? x : y;
}
*/

template <typename T>
T max(T x, T y)
{
    return (x > y) ? x : y;
}

// NOTE
// You can compile this yourself and see that it works (with the stuff in the comments)
// An instantiated function is only instantiated the first time a function call is made. 
// Further calls to the function are routed to the already instantiated function.

// ====================== OPTIONAL SECTION ENDS ======================

int main()
{
    cout << "max<int>(1, 2) = " << max<int>(1, 2) << endl; // instantiates and calls function max<int>(int, int)
    cout << "max<int>(4, 3) = " << max<int>(4, 3) << endl;    // calls already instantiated function max<int>(int, int)
    
    cout << "max<double>(1, 2) = " << max<double>(1.3, 2.7) << endl; // instantiates and calls function max<double>(double, double)

    return 0;
}
