/**
 * SkyTorch Server - C++ Extension Module
 *
 * pybind11 module definition for the server-side binary request parser.
 * Exports execute_raw_aten_inline, execute_raw_batched_aten_inline,
 * and clear_op_cache.
 *
 * Follows the pattern from backend/csrc/Module.cpp.
 */

#include <pybind11/pybind11.h>

#include "RequestParser.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "SkyTorch Server C++ Extension - Binary Request Parser";

    m.def("execute_raw_aten_inline",
        &skytorch::server::execute_raw_aten_inline,
        "Execute a raw binary execute_aten request inline",
        py::arg("data"),
        py::arg("tensor_dict"));

    m.def("execute_raw_batched_aten_inline",
        &skytorch::server::execute_raw_batched_aten_inline,
        "Execute a batch of raw binary execute_aten requests inline",
        py::arg("data"),
        py::arg("tensor_dict"));

    m.def("clear_op_cache",
        &skytorch::server::clear_op_cache,
        "Clear cached op/attr lookups (call before shutdown)");

    // Register cleanup with atexit for safe shutdown
    py::module atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function(&skytorch::server::clear_op_cache));
}
