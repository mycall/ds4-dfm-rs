#pragma once
#include <windows.h>
#include <psapi.h>
#include <cstdio>

struct n1x_working_set {
    SIZE_T old_min{}, old_max{}, max_ws{};
    DWORD old_flags{};
    bool changed{};
    bool apply() {
        constexpr size_t GiB = size_t(1) << 30;
        MEMORYSTATUSEX mem{sizeof(mem)};
        PROCESS_MEMORY_COUNTERS counters{};
        HANDLE process = GetCurrentProcess();
        if (!GlobalMemoryStatusEx(&mem) || mem.ullAvailPhys <= 4 * GiB ||
            !GetProcessMemoryInfo(process, &counters, sizeof(counters)) ||
            !GetProcessWorkingSetSizeEx(process, &old_min, &old_max, &old_flags)) {
            fprintf(stderr, "ds41: cannot establish a file-backed working set with 4 GiB free for Windows\n");
            return false;
        }
        max_ws = counters.WorkingSetSize + (mem.ullAvailPhys - 4 * GiB) / 2;
        if (!(old_flags & QUOTA_LIMITS_HARDWS_MAX_ENABLE) || max_ws < old_max) {
            if (!SetProcessWorkingSetSizeEx(process, old_min, max_ws,
                    (old_flags & ~QUOTA_LIMITS_HARDWS_MAX_DISABLE) | QUOTA_LIMITS_HARDWS_MAX_ENABLE)) {
                fprintf(stderr, "ds41: cannot bound mapped weight residency (Windows error %lu)\n", GetLastError());
                return false;
            }
            changed = true;
        }
        if (!changed) max_ws = old_max;
        return true;
    }
    ~n1x_working_set() {
        if (changed) SetProcessWorkingSetSizeEx(GetCurrentProcess(), old_min, old_max, old_flags);
    }
};
