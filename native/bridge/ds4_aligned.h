#pragma once
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#define ds4_aligned_alloc(alignment, bytes) _aligned_malloc(bytes, alignment)
#define ds4_aligned_free _aligned_free
#else
#define ds4_aligned_alloc aligned_alloc
#define ds4_aligned_free free
#endif
