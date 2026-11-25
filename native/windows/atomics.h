/* CUDA host-side counters use the same atomic storage as the C host. */
#pragma once
#include <atomic>
#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_SEQ_CST 5
#endif
#define __atomic_thread_fence(order) std::atomic_thread_fence(static_cast<std::memory_order>(order))
#define __atomic_load_n(ptr, order) std::atomic_ref(*ptr).load(static_cast<std::memory_order>(order))
#define __atomic_store_n(ptr, val, order) std::atomic_ref(*ptr).store(val, static_cast<std::memory_order>(order))
#define __atomic_fetch_add(ptr, val, order) std::atomic_ref(*ptr).fetch_add(val, static_cast<std::memory_order>(order))
#define __sync_synchronize() std::atomic_thread_fence(std::memory_order_seq_cst)
#define __atomic_compare_exchange_n(ptr, expected, desired, weak, success, failure) \
    std::atomic_ref(*ptr).compare_exchange_strong(*(expected), (desired), \
        static_cast<std::memory_order>(success), static_cast<std::memory_order>(failure))
