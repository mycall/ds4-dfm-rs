#pragma once
#include "llama-hparams.h"

// A repeated checkpoint layer supplies the state from its final visit.
inline int dfly_last_visit(const llama_hparams &hp, int checkpoint_layer) {
    const int first = hp.dsv41_repeat_at - hp.dsv41_repeat_count;
    return checkpoint_layer < first ? checkpoint_layer : checkpoint_layer + hp.dsv41_extra_layers;
}
