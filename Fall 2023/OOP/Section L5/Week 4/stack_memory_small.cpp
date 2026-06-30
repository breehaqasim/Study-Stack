#include <iostream>

void recursiveFunction(int i) 
{
    char arr[1000 * 100] {};    // 100 kB per array reqd.
    int size { sizeof(arr) };

    std::cout << "Recursive Call # " << i << std::endl;
    std::cout << "Approx. Stack Memory used = " << (size * (++i)) / 1.0E3 << " kB\n" << '\n';
    recursiveFunction(i);
}

int main() 
{
    // recursiveFunction(0);

    char arr[1000 * 1000 * 3] {};
    std::cout << static_cast<void*>(arr) << '\n';

    return 0;
}
