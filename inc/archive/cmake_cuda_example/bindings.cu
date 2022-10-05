#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gpu_library.cuh>
#include <pet2.h>

namespace py = pybind11;


PYBIND11_MODULE(gpu_library, m)
{
    m.def("multiply_with_scalar", map_array<double>);

    py::class_<Pet2>(m, "Pet2", py::dynamic_attr())
        .def(py::init<const std::string &>())
        .def("setName", &Pet2::setName)
        .def("getName", &Pet2::getName)
        .def_readwrite("name", &Pet2::name)
        .def("__repr__",
            [](const Pet2 &a) {
                return "<example.Pet2 named '" + a.name + "'>";
            }
        );
    
    py::class_<Pet>(m, "Pet", py::dynamic_attr())
    .def(py::init<const std::string &>())
    .def("setName", &Pet::setName)
    .def("getName", &Pet::getName)
    .def("register_buffer", &Pet::register_buffer)
    .def_readwrite("name", &Pet::name)
    .def_readwrite("id", &Pet::id)
    .def_readwrite("bmapped", &Pet::bmapped)
    .def("__repr__",
        [](const Pet &a) {
            return "<example.Pet named '" + a.name + "'>";
        }
    );
}
