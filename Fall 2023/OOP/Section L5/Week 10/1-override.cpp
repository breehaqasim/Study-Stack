#include<iostream>

using std::cout; using std::cin; using std::endl;
using std::string;

// Associated Reading: https://www.learncpp.com/cpp-tutorial/the-override-and-final-specifiers-and-covariant-return-types/

/* MOTIVATION
    - A derived class virtual function is only considered an override if its signature and return types match exactly. 
        - function signature = function name + parameter list + const-ness

    - What if a function was intended to be an override but you made a mistake...
        - How can you protect yourself?

    - Use "override" in derived methods.
*/

class A
{
public:
	virtual string getName1(int x) { return "A"; }  // Note the double quotes - it's a string
	virtual string getName2(int x) { return "A"; }
	virtual string getName3(int x) { return "A"; }
    virtual string getName4(int x) { return "A"; }
            string getName5(int x) { return "A"; }
    virtual string getName6(int x) { return "A"; }
};


class B : public A
{
public:
            string getName1(int x)            override { return "B"; }   // okay, function is an override of A::getName1(int)
            string getName2(int x)     const  override { return "B"; }   // compile error, function is not an override
            string getName3(double x)         override { return "B"; }   // compile error, function is not an override
            int    getName4(int x)            override { return 0; }     // compile error, function is not an override
            string getName5(int x)            override { return "B"; }   // compile error, function is not an override
            string getname6(int x)            override { return "A"; }   // compile error, function is not an override
};


/* PROBLEMS
    - getName1(): OK
    - getName2(): should be non-const
    - getName3(): parameter should be integer
    - getName4(): return type should be string
    - getName5(): base version is not virtual
    - getName6(): method names are different
*/

/* NOTE
    - Virtual functions that you intend to override, tag them with "override"

    - "override" implies "virtual", 
        so no need to write "virtual", if you've already written "override"
*/

/* NOTE (optional)
    - Check "final" from the website directly.
        - prevents from overriding a virtual function.
        - prevents derivation from a class.
*/


int main()
{
	B b{};
	A* ptrA{ &b };
	
    cout << ptrA->getName1(1) << '\n';      // output: B
	cout << ptrA->getName2(2) << '\n';      // output: B
    cout << ptrA->getName3(3) << '\n';      // output: B
    cout << ptrA->getName4(4) << '\n';      // output: 0
    cout << ptrA->getName5(5) << '\n';      // output: B

	return 0;
}


