#include<iostream>

// Write a function that tells how many times it has been called.

void tellNumCalls()
{
    int numCalls{1};

    std::cout << "Called " << ++numCalls << " times.\n";
}


int main()
{
    for(int i{}; i<5; i++)
    {
        tellNumCalls();
    }

    return 0;
}

