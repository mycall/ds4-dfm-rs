// Single-row Varloop projections. Matrices remain in their original GGUF encoding.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <new>
#include "../native/bridge/ds4_bridge.h"

struct ds4_varloop_device {
    float *x = nullptr, *out = nullptr;
    size_t nx = 0, no = 0;
};

extern "C" ds4_varloop_device *ds4_bridge_varloop_device() { return new (std::nothrow) ds4_varloop_device; }
extern "C" void ds4_bridge_varloop_close(ds4_varloop_device *d) {
    if (d) {
        cudaFree(d->x);
        cudaFree(d->out);
        delete d;
    }
}
extern "C" void *ds4_bridge_varloop_upload(const uint8_t *bytes, size_t n) {
    void *p = nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess)
        return nullptr;
    if (cudaMemcpy(p, bytes, n, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p);
        return nullptr;
    }
    return p;
}
extern "C" void ds4_bridge_varloop_free(void *p) { cudaFree(p); }

static cudaError_t reserve(float *&ptr, size_t &capacity, size_t need) {
    if (capacity >= need)
        return cudaSuccess;
    float *next = nullptr;
    cudaError_t err = cudaMalloc(&next, need * sizeof(float));
    if (err == cudaSuccess) {
        cudaFree(ptr);
        ptr = next;
        capacity = need;
    }
    return err;
}

template <unsigned kind>
__global__ static void project(const uint8_t *weights, size_t first, int width, const float *x, float *out) {
    constexpr int threads = 128;
    const size_t row = first + blockIdx.x;
    const size_t base = row * width;
    float sum = 0;
    for (int i = threadIdx.x; i < width; i += threads) {
        float v;
        if constexpr (kind == 0)
            v = reinterpret_cast<const float *>(weights)[base + i];
        if constexpr (kind == 1)
            v = __half2float(reinterpret_cast<const __half *>(weights)[base + i]);
        if constexpr (kind == 30)
            v = __uint_as_float(uint32_t(reinterpret_cast<const uint16_t *>(weights)[base + i]) << 16);
        if constexpr (kind == 8) {
            const uint8_t *block = weights + ((base + i) / 32) * 34;
            v = __half2float(*reinterpret_cast<const __half *>(block)) *
                reinterpret_cast<const int8_t *>(block + 2)[i % 32];
        }
        sum = fmaf(v, x[i], sum);
    }
    __shared__ float partial[threads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = partial[0];
}
extern "C" int ds4_bridge_varloop_mv(ds4_varloop_device *d, const void *p, size_t first, uint32_t kind,
                                     int width, int rows, const float *x, float *out) {
    cudaError_t err = reserve(d->x, d->nx, width);
    if (err != cudaSuccess)
        return err;
    err = reserve(d->out, d->no, rows);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpy(d->x, x, width * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return err;
    const auto *w = static_cast<const uint8_t *>(p);
    switch (kind) {
    case 0:
        project<0><<<rows, 128>>>(w, first, width, d->x, d->out);
        break;
    case 1:
        project<1><<<rows, 128>>>(w, first, width, d->x, d->out);
        break;
    case 8:
        project<8><<<rows, 128>>>(w, first, width, d->x, d->out);
        break;
    case 30:
        project<30><<<rows, 128>>>(w, first, width, d->x, d->out);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    err = cudaGetLastError();
    return err == cudaSuccess ? cudaMemcpy(out, d->out, rows * sizeof(float), cudaMemcpyDeviceToHost) : err;
}
