#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct layer {

    int num_inputs, num_outputs;

    double *outputs;

    struct layer *prev;
    struct layer *next;

    double **weights;

    double *biases;

    double *deltas;

} layer_t;

double sigmoid(double x);

double sigmoidPrime(double x);

layer_t *layer_create();

int layer_init(layer_t *layer, int num_outputs, layer_t *prev);

void layer_free(layer_t *layer);

void layer_compute_outputs(layer_t const *layer);

void layer_compute_deltas(layer_t const *layer);

void layer_update(layer_t const *layer, double l_rate);