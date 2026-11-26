#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// The slot owns the event and its OVERLAPPED storage until the read completes.
struct ds41_expert_read {
    OVERLAPPED operation{};
    HANDLE file = INVALID_HANDLE_VALUE;
    size_t required = 0;

    ds41_expert_read() { operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr); }
    ds41_expert_read(const ds41_expert_read &) = delete;
    ds41_expert_read & operator=(const ds41_expert_read &) = delete;
    ~ds41_expert_read() {
        finish();
        if (operation.hEvent) CloseHandle(operation.hEvent);
    }

    bool start(HANDLE handle, uint64_t offset, void *buffer, DWORD length, size_t minimum) {
        if (!operation.hEvent || file != INVALID_HANDLE_VALUE) return false;
        const HANDLE event = operation.hEvent;
        operation = {};
        operation.hEvent = event;
        operation.Offset = static_cast<DWORD>(offset);
        operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
        if (!ReadFile(handle, buffer, length, nullptr, &operation) && GetLastError() != ERROR_IO_PENDING) {
            return false;
        }
        file = handle;
        required = minimum;
        return true;
    }

    bool finish() {
        if (file == INVALID_HANDLE_VALUE) return false;
        DWORD bytes = 0;
        const bool ok = GetOverlappedResult(file, &operation, &bytes, TRUE) && bytes >= required;
        file = INVALID_HANDLE_VALUE;
        return ok;
    }
};
#else
struct ds41_expert_read {
    bool finish() { return false; }
};
#endif
