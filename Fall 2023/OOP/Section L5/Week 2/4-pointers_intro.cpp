#include<iostream>

int main()
{
    double x{3.14};

    // The address-of operator (&)
    std::cout << "x = " << x << std::endl;        // The value of x
    std::cout << "&x = " << &x << std::endl;      // The address at which x is created.

    std::cin.get();




    // The dereference operator (*)
    std::cout << "*(&x) = " << *(&x) << std::endl;

    std::cin.get();




    // =====================================================
    // POINTERS
    // =====================================================
    
    // Declaration; such pointers are called WILD POINTERS.
    double* ptr1;

    double *ptr2;

    /* NOTE
        - The first style of writing (double*) is preferred over the latter (double *)
        
        - Note, that the asterisk here isn't a dereference operator.

        - Avoid wild pointers.
            - Dereferencing them will result in an undefined behavior.
    */

    // This is a null pointer btw. 
    double* ptr3 {};

    // Better to this though when creating null pointers.
    double* ptr4 { nullptr };

    // We will talk more about null pointers later.


    double* x_ptr {&x};
    std::cout << "*x_ptr = " << *x_ptr << std::endl;

    std::cin.get();




    // Importance of Pointer types.

    double pi{3.142};
    int* pi_ptr{&pi};   // Will not work.


    return 0;
}
