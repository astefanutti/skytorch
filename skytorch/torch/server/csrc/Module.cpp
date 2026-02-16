/**
 * SkyTorch Server - C++ Extension Module
 *
 * pybind11 module definition for the server-side binary request parser.
 * Exports TensorStore, execute_raw_aten_inline, execute_raw_batched_aten_inline,
 * and clear_op_cache.
 *
 * Follows the pattern from backend/csrc/Module.cpp.
 */

#include <pybind11/pybind11.h>

#include "RequestParser.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "SkyTorch Server C++ Extension - Binary Request Parser";

    py::class_<skytorch::server::TensorStore>(m, "TensorStore")
        .def(py::init<>())
        .def("get", &skytorch::server::TensorStore::get_python,
            "Get tensor as Python object (GIL required)",
            py::arg("id"))
        .def("set", &skytorch::server::TensorStore::set_python,
            "Set tensor from Python object (GIL required)",
            py::arg("id"), py::arg("tensor"))
        .def("erase", &skytorch::server::TensorStore::erase,
            "Erase tensor by ID",
            py::arg("id"))
        .def("__contains__", &skytorch::server::TensorStore::contains,
            py::arg("id"))
        .def("__len__", &skytorch::server::TensorStore::size)
        .def("clear", &skytorch::server::TensorStore::clear);

    m.def("execute_raw_aten_inline",
        &skytorch::server::execute_raw_aten_inline,
        "Execute a raw binary execute_aten request inline",
        py::arg("data"),
        py::arg("store"));

    m.def("execute_raw_batched_aten_inline",
        &skytorch::server::execute_raw_batched_aten_inline,
        "Execute a batch of raw binary execute_aten requests inline",
        py::arg("data"),
        py::arg("store"));

    m.def("clear_op_cache",
        &skytorch::server::clear_op_cache,
        "Clear cached op/attr lookups (call before shutdown)");

    // Register cleanup with atexit for safe shutdown
    py::module atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function(&skytorch::server::clear_op_cache));
}
