#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct layer {

    int num_inputs, num_outputs; // Layer dimensions

    double *outputs; // Calculated output values

    struct layer *prev; // Pointer to next layer
    struct layer *next; // Pointer to previous layer

    double **weights; // Matrix of Weights

    double *biases; // Biases for each node

    double *deltas; // Deltas for back propagation

} layer_t;

double sigmoid(double x);

double sigmoidPrime(double x);

layer_t *layer_create();

int layer_init(layer_t *layer, int num_outputs, layer_t *prev);

void layer_free(layer_t *layer);

void layer_compute_outputs(layer_t const *layer);

void layer_compute_deltas(layer_t const *layer);

void layer_update(layer_t const *layer, double l_rate);