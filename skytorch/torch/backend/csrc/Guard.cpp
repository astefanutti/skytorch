/**
 * SkyTorch PyTorch Backend - Device Guard Implementation
 *
 * This module implements the DeviceGuardImplInterface for the SkyTorch backend.
 * The device guard manages device context switching for operations.
 */

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace skytorch {

// Forward declaration for get_method (defined in Module.cpp)
py::object get_method(const std::string& name);

// Forward declarations for device management (defined in Module.cpp)
c10::DeviceIndex device_count();
c10::DeviceIndex current_device();
void set_device(c10::DeviceIndex device);
c10::DeviceIndex exchange_device(c10::DeviceIndex device);

/**
 * SkyTorch Device Guard Implementation
 *
 * Implements the DeviceGuardImplInterface for managing SkyTorch device context.
 * Calls Python driver methods with GIL acquisition for device operations.
 */
struct GuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    GuardImpl() = default;

    explicit GuardImpl(c10::DeviceType t) {
        TORCH_INTERNAL_ASSERT(t == static_type);
    }

    // Device management

    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const override {
        TORCH_INTERNAL_ASSERT(d.is_privateuseone());
        // Use C++ exchange_device() to update both C++ thread-local and Python
        auto old_device_index = exchange_device(d.index());
        return c10::Device(static_type, old_device_index);
    }

    c10::Device getDevice() const override {
        return c10::Device(static_type, current_device());
    }

    void setDevice(c10::Device d) const override {
        TORCH_INTERNAL_ASSERT(d.is_privateuseone());
        // Use C++ set_device() to update both C++ thread-local and Python
        set_device(d.index());
    }

    void uncheckedSetDevice(c10::Device d) const noexcept override {
        if (d.index() >= 0 && d.index() < device_count()) {
            try {
                // Use C++ set_device() to update both C++ thread-local and Python
                set_device(d.index());
            } catch (...) {
                // Ignore errors in unchecked version
            }
        }
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        return device_count();
    }

    // stream management

    c10::Stream getStream(c10::Device d) const noexcept override {
        py::gil_scoped_acquire acquire;
        try {
            auto stream_id =
                get_method("get_stream")(d.index()).cast<c10::StreamId>();
            return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
        } catch (...) {
            return c10::Stream(c10::Stream::DEFAULT, d);
        }
    }

    c10::Stream getDefaultStream(c10::Device d) const override {
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
        py::gil_scoped_acquire acquire;
        auto stream_id =
            get_method("get_new_stream")(d.index(), priority).cast<c10::StreamId>();
        return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
    }

    c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
        return getNewStream(d, isHighPriority ? -1 : 0);
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        py::gil_scoped_acquire acquire;
        try {
            auto previous_stream_id =
                get_method("exchange_stream")(s.id(), s.device().index())
                    .cast<c10::StreamId>();
            return c10::Stream(c10::Stream::UNSAFE, s.device(), previous_stream_id);
        } catch (...) {
            return s;
        }
    }

    void synchronizeStream(const c10::Stream& stream) const override {
        py::gil_scoped_acquire acquire;
        get_method("synchronize_stream")(stream.id(), stream.device().index());
    }

    // Event management

    void createEvent(
        void** event,
        const c10::DeviceIndex device_index,
        const c10::EventFlag flag) const {
        py::gil_scoped_acquire acquire;
        auto event_id =
            get_method("create_event")(device_index, static_cast<int64_t>(flag))
                .cast<int64_t>();
        *event = reinterpret_cast<void*>(event_id);
    }

    void record(
        void** event,
        const c10::Stream& stream,
        const c10::DeviceIndex device_index,
        const c10::EventFlag flag) const override {
        // Create event if needed
        if (*event == nullptr) {
            createEvent(event, device_index, flag);
        }
        // Recording is a no-op for now
    }

    void block(void* event, const c10::Stream& stream) const override {
        // Blocking is a no-op for synchronous operations
    }

    bool queryEvent(void* event) const override {
        // Events are always complete for synchronous operations
        return true;
    }

    void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override {
        // No-op - event IDs don't need explicit cleanup
    }
};

// Register the guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardImpl);

// Registration function called from Module.cpp
void register_guard() {
    // Guard is registered via C10_REGISTER_GUARD_IMPL macro
    // This function is kept for explicit registration if needed
}

}  // namespace skytorch
