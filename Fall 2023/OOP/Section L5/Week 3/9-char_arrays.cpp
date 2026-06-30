#include<iostream>

int main()
{
    char charArr[]{ "bayes' theorem" };
    std::cout << "charArr = " << charArr << '\n';

    // Extension to the concept discussed above.
    char alph{ 'a' };

    // What will be the output here?
    std::cout << "alph = " << alph << '\n';

    std::cin.get();



    // What will be the output here?
    std::cout << "&alph = " << &alph << '\n';

    std::cin.get();



    // ====================================================================
    // So then how will you print the address of a char type data?
    // Before that though...
    
    /* VOID POINTERS

        - Can point at a variable of any data type.
    */

    void* void_ptr{ nullptr };

    int a{ 5 };
    double b{ 2.99E8 };
    char c{ 'h' };

    int A[]{ 1, 3, -2 };
    double B[]{ 1.0, 3.14, -273.15 };
    char C[]{ "absolute zero" };

    void_ptr = &a;
    std::cout << "void_ptr pointing to a = " << void_ptr << std::endl;

    // void_ptr = &b;
    // std::cout << "void_ptr pointing to b = " << void_ptr << std::endl;

    // void_ptr = &c;
    // std::cout << "void_ptr pointing to c = " << void_ptr << std::endl << std::endl;


    // void_ptr = A;
    // std::cout << "void_ptr pointing to A = " << void_ptr << std::endl;

    // void_ptr = B;
    // std::cout << "void_ptr pointing to B = " << void_ptr << std::endl;

    // void_ptr = C;
    // std::cout << "void_ptr pointing to C = " << void_ptr << std::endl;




    // ====================================================================
    // LIMITATIONS of void*

    // // 1- Cannot dereference void pointers.
    // // void* does not have an associated data type. 
    // std::cout << "Value of a = " << *void_ptr << std::endl;

    // In order to dereference, cast.
    std::cout << "Value of a = " << *( static_cast<int*>(void_ptr) ) << std::endl;
    std::cin.get();

    // 2- Cannot perform pointer arithmetic with void*
    void_ptr = A;
    std::cout << "The address of A[0] = " << void_ptr << std::endl;
    std::cout << "(A + 1) The address of A[1] = " << (A + 1) << std::endl;
    std::cout << "(void_ptr + 1) The address of A[1] = " << (void_ptr + 1) << std::endl;      // Incorrect output
    std::cout << "(int*)void_ptr + 1 = " << (int*)void_ptr + 1 << std::endl;          // But after casting to int*, pointer arithmetic makes sense.

    


    // ====================================================================
    // PRACTICE
    char str1[] { "neural nets" };

    std::cout << "str1 =      " << str1 << std::endl;  // neural nets

    std::cout << "*str1 =     " << *str1 << std::endl;// n
    std::cout << "str1 + 2 =  " << str1 + 2 << std::endl; // ural nets
    
    char character {'m'};
    std::cout << "character =     " << character << std::endl;
    std::cout << "&character =    " << &character << std::endl;
    std::cout << "*(&character) = " << *(&character) << std::endl;




    return 0;
}