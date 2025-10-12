#include<iostream>

class A {
  public:
    void func() {
      std::cout << "func" << std::endl;
    }
};

class B {
  public:
    A* func() {
      A* a = new A();
      return a;
    }
};

class C {
    public:
        B func() {
            B b;
            return b;
        }
};

int main() {
    C c1;
    auto cptr = &c1;
    cptr->func().func()->func();
}