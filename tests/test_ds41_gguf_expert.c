/* Exercise a real IQ2_XXS expert from the Unsloth DS41F GGUF. Only one
 * expert matrix enters CUDA. The native scalar dot verifies every output row. */
#include "../ds4.c"
#include "../cuda/mmq/ds4_mmq.h"

#define CUDA_OK(call) do { \
    cudaError_t rc = (call); \
    if (rc != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); return 1; } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 2) { return 2; }
    ds4_model model;
    model_open(&model, argv[1], false, false);
    const ds4_tensor *t = model_find_tensor(&model, "blk.2.ffn_down_exps.weight");
    if (!t || t->type != DS4_TENSOR_IQ2_XXS || t->ndim != 3 ||
        t->dim[0] != 2304 || t->dim[1] != 5120 || t->dim[2] != 384) {
        fprintf(stderr, "expected the Unsloth DS41F UD-IQ2_XXS layer 2 down projection\n");
        return 1;
    }
    int k = (int)t->dim[0], m = (int)t->dim[1];
    size_t row_bytes = k / QK_K * sizeof(block_iq2_xxs);
    size_t expert_bytes = row_bytes * m;
    unsigned expert = (unsigned)t->dim[2] - 1;
    if (expert_bytes * t->dim[2] != t->bytes) { return 1; }
    const uint8_t *weights = (const uint8_t *)tensor_data(&model, t) + expert * expert_bytes;
    float *x = xcalloc(k, sizeof(float)), *out = xmalloc(m * sizeof(float));
    void *w_gpu, *x_gpu, *id_gpu, *out_gpu;
    if (ds4_mmq_init(0)) { return 1; }
    CUDA_OK(cudaMalloc(&w_gpu, expert_bytes));
    CUDA_OK(cudaMalloc(&x_gpu, k * sizeof(float)));
    CUDA_OK(cudaMalloc(&out_gpu, m * sizeof(float)));
    CUDA_OK(cudaMalloc(&id_gpu, sizeof(int32_t)));
    CUDA_OK(cudaMemset(id_gpu, 0, sizeof(int32_t)));
    CUDA_OK(cudaMemcpy(w_gpu, weights, expert_bytes, cudaMemcpyHostToDevice));
    const int cols[] = {0, k / 2 + 1, k - 1};
    for (unsigned test = 0; test < sizeof(cols) / sizeof(*cols); ++test) {
        /* 127 quantizes exactly to Q8_1 with a unit scale. */
        x[cols[test]] = 127.0f;
        CUDA_OK(cudaMemcpy(x_gpu, x, k * sizeof(float), cudaMemcpyHostToDevice));
        if (ds4_mmq_iq2_xxs_moe_vec(w_gpu, x_gpu, id_gpu, out_gpu, m, k, 1, 1, 1, NULL)) {
            return 1;
        }
        CUDA_OK(cudaMemcpy(out, out_gpu, m * sizeof(float), cudaMemcpyDeviceToHost));
        /* vec_dot_iq2_xxs_q8_1 truncates the signed integer sum / 8
         * before applying the half scale. Recover the integer coefficient
         * independently from the scalar decoder with an eight-unit input. */
        x[cols[test]] = 8.0f;
        for (int row = 0; row < m; ++row) {
            const block_iq2_xxs *blocks = (const block_iq2_xxs *)(weights + row * row_bytes);
            const float scale = f16_to_f32(blocks[cols[test] / QK_K].d);
            const float decoded = ds4_vec_dot_iq2_xxs_f32(k, blocks, x);
            const int coefficient = scale ? (int)lround((double)decoded / scale) : 0;
            const float ref = scale * ((coefficient * 127) / 8);
            if (out[row] != ref) {
                fprintf(stderr, "row %d column %d GPU %.9g reference %.9g\n", row, cols[test], out[row], ref);
                return 1;
            }
        }
        x[cols[test]] = 0.0f;
    }
    CUDA_OK(cudaFree(w_gpu));
    CUDA_OK(cudaFree(x_gpu));
    CUDA_OK(cudaFree(id_gpu));
    CUDA_OK(cudaFree(out_gpu));
    free(x);
    free(out);
    model_close(&model);
    printf("PASS IQ2_XXS expert %u, %zu bytes, %d output values\n", expert, expert_bytes, m * 3);
    return 0;
}
