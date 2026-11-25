/* The N1X build admits a single local engine. The distributed transport
 * uses POSIX sockets and descriptor passing. Reject it before model open. */
#include "ds4_distributed.h"

static int unsupported(char *err, size_t len) {
    if (len) {
        snprintf(err, len, "distributed inference is unavailable in the native Windows build");
    }
    return 1;
}

int ds4_dist_prepare_engine_options(const ds4_dist_options *opt, ds4_engine_options *engine,
                                    char *err, size_t len) {
    if (opt && (opt->role != DS4_DISTRIBUTED_NONE || opt->replay_check || opt->layers.set)) {
        return unsupported(err, len);
    }
    if (opt && engine) engine->distributed = *opt;
    return 0;
}

int ds4_dist_session_create(ds4_dist_session **out, ds4_engine *engine, const ds4_dist_options *opt,
                            ds4_session *owner, int ctx_size, char *err, size_t len) {
    *out = NULL;
    return unsupported(err, len);
}
void ds4_dist_session_free(ds4_dist_session *d) {
    (void)d;
}
int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t len) {
    unsupported(err, len);
    return -1;
}
int ds4_dist_session_sync(ds4_dist_session *d, ds4_session *owner, const ds4_tokens *checkpoint,
                          const ds4_tokens *prompt, float *logits, char *err, size_t len) {
    return unsupported(err, len);
}
int ds4_dist_session_eval(ds4_dist_session *d, ds4_session *owner, const ds4_tokens *checkpoint,
                          int token, float *logits, char *err, size_t len) {
    return unsupported(err, len);
}
int ds4_dist_session_save_payload(ds4_dist_session *d, ds4_session *owner, FILE *fp, char *err, size_t len) {
    return unsupported(err, len);
}
int ds4_dist_session_load_payload(ds4_dist_session *d, ds4_session *owner, FILE *fp,
                                  uint64_t bytes, char *err, size_t len) {
    return unsupported(err, len);
}
