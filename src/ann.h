#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random.h"
#include "layer.h"

typedef struct ann {

    layer_t *input_layer;
    layer_t *output_layer;

} ann_t;

ann_t *ann_create(int num_layers, int *layer_outputs);

void ann_free(ann_t *ann);

void ann_predict(ann_t const *ann, double const *inputs);

void ann_train(ann_t *ann, double *inputs, double *targets, double l_rate);