#include <iostream>

struct Pet2 {

    Pet2(const std::string &name) : name(name) { }
    void setName(const std::string &name_);
    const std::string &getName() const { return name; }

    std::string name;
};