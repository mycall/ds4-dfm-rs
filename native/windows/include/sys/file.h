#pragma once
#include <unistd.h>
#define LOCK_EX 2
#define LOCK_NB 4
static inline int flock(int fd, int op) {
    OVERLAPPED ov = {0};
    if (op != (LOCK_EX | LOCK_NB)) { errno = EINVAL; return -1; }
    if (LockFileEx((HANDLE)_get_osfhandle(fd), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                   0, MAXDWORD, MAXDWORD, &ov)) return 0;
    errno = GetLastError() == ERROR_LOCK_VIOLATION ? EWOULDBLOCK : EIO;
    return -1;
}
