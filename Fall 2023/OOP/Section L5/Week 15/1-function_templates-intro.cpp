#include<iostream>
using std::cout;    // allows abbreviating "std::cout" to "cout"
using std::cin;     // allows abbreviating "std::cin" to "cin"
using std::endl;    // allows abbreviating "std::endl" to "endl"

// Associated Reading: https://www.learncpp.com/cpp-tutorial/function-templates/

/* DISCUSSION
    - Writing multiple overloads for a function that does the same thing, can create some problems.

        1- The implementation would be similar, if not the same, and you generally want to try NOT REPEATING YOURSELF when coding.

        2- Maintenance issues - you have multiple functions to take care of now. 

        3- You may not include all possible implementations, so that could be a problem as well.

    - If you could have a single function that could accept different types, that'd be great.
        - The answer lies in TEMPLATES.
*/

int max(int x, int y)
{
    // Ternary operator (optional to know).
    // return (x > y) ? x : y;

    if(x > y) {
        return x;
    }
    else {
        return y;
    }
}


double max(double x, double y)
{
    // return (x > y) ? x : y;

    if(x > y) {
        return x;
    }
    else {
        return y;
    }
}


/* DISCUSSION
    - When we create our function template, we use placeholder types (also called template types) for, 
        - any parameter types, 
        - return types, or 
        - types used in the function body that we want to be specified later.
*/

template <typename T>   // this is the template parameter declaration, ...
                        // ... lets the compiler knows that we're creating a template.

T max(T x, T y)     // and this is the function template definition for max<T>
{
    // return (x > y) ? x : y;

    if(x > y) {
        return x;
    }
    else {
        return y;
    }
}

// The compiler can use the template to generate as many overloaded functions (even classes!) as needed, each using different actual types!

// Now YOU only need to maintain the template.

// Templates can work with types that didn't exist when the templates were written.


/* NOTE
    - Each template function (or template class) definition needs its own template parameter declaration.
    - So if you remove the template parameter declaration, the following function template definition won't work. 
    
    - In this context, using "class" or "typename" means the same.
        - "typename" is better because we're saying that there could be anything here - fundamental types or class types.

    - We call these function template definitions, max<T> and min<T>, since they're based on just one template type.
*/

template <class T>      // "class" instead of "typename". "typename" is preferred.
T min(T x, T y)
{
    // return (x < y) ? x : y;

    if(x > y) {
        return x;
    }
    else {
        return y;
    }
}
