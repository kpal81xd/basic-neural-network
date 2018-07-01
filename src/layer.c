#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "random.h"

typedef struct layer {

    int num_inputs, num_outputs;

    double *outputs;

    struct layer *prev;
    struct layer *next;

    double **weights;

    double *biases;

    double *deltas;

} layer_t;

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidPrime(double x){
    return x * (1.0 - x);
}

layer_t *layer_create() {
    layer_t *layer = malloc(sizeof(layer_t));
    return layer;
}

int layer_init(layer_t *layer, int num_outputs, layer_t *prev){
    layer->num_inputs = 0;
    layer->num_outputs = num_outputs;
    layer->prev = NULL;
    layer->next = NULL;

    if (prev){
        layer->prev = prev;
        prev->next = layer;

        layer->num_inputs = prev->num_outputs;
    }

    layer->outputs = calloc(layer->num_outputs, sizeof(double));
    layer->biases = calloc(layer->num_outputs, sizeof(double));
    layer->deltas = calloc(layer->num_outputs, sizeof(double));

    layer->weights = calloc(layer->num_inputs, sizeof(double*));
    for (int i = 0; i < layer->num_inputs;i++){
        layer->weights[i] = calloc(layer->num_outputs, sizeof(double));
    }

    for (int y = 0; y < layer->num_inputs; y++){
        for (int x = 0; x < layer->num_outputs; x++){
            layer->weights[y][x] = 0;//get_random();
        }    
        
    } 

    return 1;
}

void layer_free(layer_t *layer){
    free(layer->outputs);
    free(layer->biases);
    free(layer->deltas);
    for (int i = 0; i < layer->num_inputs;i++){
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer);
}

void layer_compute_outputs(layer_t const *layer){
    for (int x = 0; x < layer->num_outputs; x++){
        double total = 0;
        for (int y = 0; y < layer->num_inputs; y++){   
            total += layer->prev->outputs[y] * layer->weights[y][x];
        }
        layer->outputs[x] = sigmoid(layer->biases[x] + total);
    }  
}

void layer_compute_deltas(layer_t const *layer){
    for (int y = 0; y < layer->next->num_inputs; y++){
        double total = 0;
        for (int x = 0; x < layer->next->num_outputs; x++){
            total += layer->next->deltas[x] * layer->next->weights[y][x];
        }
        layer->deltas[y] = sigmoidPrime(layer->outputs[y]) * total;       
    }
}

void layer_update(layer_t const *layer, double l_rate){
    for (int y = 0; y < layer->num_inputs; y++){
        for (int x = 0; x < layer->num_outputs; x++){
            layer->weights[y][x] += l_rate * layer->prev->outputs[y] * layer->deltas[x];
            printf("W: %lf ",layer->weights[y][x]);
        }
        printf("\n");      
    }
    printf("\n");

    for (int i = 0; i < layer->num_outputs; i++){
        layer->biases[i] += l_rate * layer->deltas[i];
    }
}
