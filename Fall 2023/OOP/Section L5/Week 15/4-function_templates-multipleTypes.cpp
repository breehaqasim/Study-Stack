#include<iostream>

// Associated Reading: https://www.learncpp.com/cpp-tutorial/function-templates-with-multiple-template-types/


// I have the following function template.

template <typename T>
T max(T x, T y)
{
    std::cout << "Template overload" << std::endl;
    return (x > y) ? x : y;
}

int main()
{
    // Now I want to pass two different data types, an integer and a double. 
    // These are some things that I could do.

    std::cout << "max<double>(2, 3.5) = " << max<double>(2, 3.5) << std::endl;  // int gets upgraded to double
    
    std::cout << "max<int>(2, 3.5) = " << max<int>(2, 3.5) << std::endl;    // double gets downgraded to int
    
    std::cout << std::endl;


    /* OTHER SOLUTIONS TO SOLVE THIS PROBLEM
    
        1- static_cast<>() one of the variables to the same type as the other.
            - If you are already specifying the template type (which you should), then this becomes redundant.

        2- Provide a non-template function with the appropriate types.
            - We want to work with templates, so what else can we do?

        3- Use multiple template type parameters.
            - Topic of this lesson.
    */

    return 0;
}


////////////////////////////////////////////////////////////////
/* BEFORE WE PROCEED...
    - Recall automatic type conversion.
    - What will be the output?

    std::cout << 10 / 4;
    std::cout << 10 / 4.0;
    std::cout << 10.0 / 4;
*/


/* DISCUSSION
    - OUR REQUIREMENT: Pass in variables of different data types.
    - OUR LIMITATION: We are working with a single template parameter.

    - POSSIBLE SOLUTION: Introduce another template parameter!
*/

template <typename T, typename U> // We're using two template type parameters named T and U
auto max(T x, U y)                 // x can resolve to type T, and y can resolve to type U
{
    // Have to use this operator.
    auto var = (x > y) ? x : y;
    
    return var;
    // The following alternative won't work here. 
    // if(x > y){
    //     return x;
    // }
    // else{
    //     return y;
    // }
}

int main()
{
    std::cout << max<int, double>(2, 3.5) << '\n';

    /* DISCUSSION
        - The function call problem seems to have been resolved.

        - But what should be the return type of our function?
            - For this, we need to understand the working of the function.
            - Assume, the function call was:    max<int, int>(2, 4)
            - What happens within the function in terms of automatic type conversion? (everything stays int)
            - Return type would be int.

            - Now, consider our current case:   max<int, double>(2, 3.5)
            - What happens within the function in terms of automatic type conversion? (int automatically upgrades to double)
            - Return type would be double.

        - How to manage different return types?
        - New keyword: "auto"
            - Let the compiler determine the return type at compile time.
    */
    return 0;
}

// NOTE: T and U can be the same data type as well, i.e. <int, int>.
// Can refer to the following link for more information: 
    // https://stackoverflow.com/questions/41536407/auto-function-with-if-statement-wont-return-a-value
