#pragma once
#include <cstddef>
#include <cstdint>

enum class n1x_snapshot_class : uint8_t { recent, frequent };
struct n1x_snapshot_history;
extern "C" {
int32_t n1x_seed_quota(size_t layer, int32_t slots, size_t layers, size_t stride);
n1x_snapshot_history * n1x_history_new();
void n1x_history_free(n1x_snapshot_history *);
n1x_snapshot_class n1x_victim_class(const n1x_snapshot_history *, size_t recent_bytes, size_t budget);
void n1x_record_eviction(n1x_snapshot_history *, const uint8_t * key, size_t bytes, n1x_snapshot_class cls, size_t budget);
bool n1x_revisit(n1x_snapshot_history *, const uint8_t * key, size_t budget);
}
