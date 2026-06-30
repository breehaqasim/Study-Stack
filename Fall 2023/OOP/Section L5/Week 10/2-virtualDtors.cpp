#include<iostream>

class Base
{
public:
    ~Base() { std::cout << "Base dtor\n"; }
};


class Derived : public Base
{
public:
    ~Derived() { std::cout << "Derived dtor\n"; }
};

int main()
{
    Derived d1;

    // // Need to make Base::~Base() virtual for the delete to work successfully.
    // Base* pBase = new Derived;
    
    // delete pBase;
    
    return 0;
}
