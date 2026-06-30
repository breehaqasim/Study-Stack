#include<iostream>
#include<array>

/* DISCUSSION

    PROBLEM SETUP
    -------------
    - In our Array class below, we want an array on the stack memory.

    - Size has to be given at compile time (hence it will be some constant, say 5).
        - The getSize()'s body would then be just { return 5; }.

    - The data type of the array would also be fixed - let's say int for our example.

    - In order to introduce some generic-ness in our class, we can make use of templates.

    TEMPLATE CLASSES
    ----------------
    - To generic-ify the data type of the array, introduce a template parameter, and use that as a placeholder for the array's data type.
        - Similar to how you would do in the case of function parameters in function templates.

    - However, template parameters are not limited to being just placeholders for datatypes.
        - Enter "Non-type template parameters".

    - As mentioned above, because the array is on the stack memory, I want to specify the size at compile time.
        - So it has to be a literal integer (like 5) or a const variable.

    - This can be templated as well as shown in the code below. 
        - When the compiler creates a copy of this template class, the function body of getSize() will be { return 5; } (assuming N = 5).
        - The function would still return a constant.
        - This is different than { return size; } where size is some variable. N is a template parameter.

    - The built-in C++ STL library is based on templates (STL: Standard Template Library).
        - You can Ctrl+click (on Windows) the "array" in #include<array>.
        - Observe the std namespace and templated classes.
*/

// In order to get rid of warnings during compilation, you can replace "int N" with "unsigned long long N" or "std::size_t N"
template<typename T, int N> 
class Array
{
private:
    T m_data[N];

public:
    int getSize() const { return N; }
};

int main()
{
    // Our class
    Array<double, 3> arr1;
    std::cout << arr1.getSize();

    // Built-in class
    std::array<double*, 10> arr;

    return 0;
}

