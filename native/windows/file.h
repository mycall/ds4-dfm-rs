/* Win32 adapters for the native engine's large-file I/O. */
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

typedef int64_t ds4_file_offset;
#define fseeko _fseeki64
#define ftello _ftelli64

/* Windows rename() cannot replace an existing checkpoint. */
static inline int ds4_install_file(const char *from, const char *to) {
    if (MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING)) return 0;
    errno = EIO;
    return -1;
}

static inline int ds4_mkstemp(char *path) {
    size_t len = strlen(path) + 1;
    if (_mktemp_s(path, len) != 0) return -1;
    return _open(path, _O_CREAT | _O_EXCL | _O_BINARY | _O_RDWR | _O_NOINHERIT,
                 _S_IREAD | _S_IWRITE);
}

/* Open a private delete-on-close file in the requested directory. */
static inline int ds4_temp_fd(const char *dir) {
    char path[MAX_PATH];
    for (int i = 0; i < 16; ++i) {
        int n = snprintf(path, sizeof(path), "%s/dfmXXXXXX", dir);
        if (n < 0 || (size_t)n >= sizeof(path)) { errno = ENAMETOOLONG; return -1; }
        if (_mktemp_s(path, sizeof(path))) return -1;
        int fd;
        errno_t err = _sopen_s(&fd, path, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY |
                               _O_NOINHERIT | _O_TEMPORARY, _SH_DENYRW, _S_IREAD | _S_IWRITE);
        if (!err) return fd;
        if (err != EEXIST) { errno = err; return -1; }
    }
    errno = EEXIST;
    return -1;
}

static inline FILE *ds4_temp_stream(void) {
    char dir[MAX_PATH];
    DWORD n = GetTempPathA(sizeof(dir), dir);
    if (!n || n >= sizeof(dir)) return NULL;
    int fd = ds4_temp_fd(dir);
    if (fd < 0) return NULL;
    FILE *fp = _fdopen(fd, "w+b");
    if (!fp) _close(fd);
    return fp;
}
