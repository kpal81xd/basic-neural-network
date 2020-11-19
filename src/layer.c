#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "layer.h"
#include "random.h"

// Sigmoid activation function
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
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

    // Allocate memory to arrays and matrices
    layer->outputs = calloc(layer->num_outputs, sizeof(double));
    if (!layer->outputs) return 1;
    layer->biases = calloc(layer->num_outputs, sizeof(double));
    if (!layer->biases) return 1;
    layer->deltas = calloc(layer->num_outputs, sizeof(double));
    if (!layer->deltas) return 1;

    layer->weights = calloc(layer->num_inputs, sizeof(double*));
    if (!layer->weights) return 1;
    for (int i = 0; i < layer->num_inputs;i++){
        layer->weights[i] = calloc(layer->num_outputs, sizeof(double));
        if (!layer->weights[i]) return 1;
    }

    // Allocate random weights
    for (int y = 0; y < layer->num_inputs; y++){
        for (int x = 0; x < layer->num_outputs; x++){
            layer->weights[y][x] = get_random();
        }    
        
    } 

    return 0;
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

// Calcuate layer outputs
// O = sigmoid((O_prev x W) + B)
void layer_compute_outputs(layer_t const *layer){
    for (int x = 0; x < layer->num_outputs; x++){
        double total = 0;
        for (int y = 0; y < layer->num_inputs; y++){   
            total += layer->prev->outputs[y] * layer->weights[y][x];
        }
        layer->outputs[x] = sigmoid(layer->biases[x] + total);
    }  
}

// Calcuate layer deltas
// D = sigmoid'(O) x W x D_next
void layer_compute_deltas(layer_t const *layer){
    for (int y = 0; y < layer->next->num_inputs; y++){
        double total = 0;
        for (int x = 0; x < layer->next->num_outputs; x++){
            total += layer->next->deltas[x] * layer->next->weights[y][x];
        }
        layer->deltas[y] = sigmoidPrime(layer->outputs[y]) * total;       
    }
}

// Update layer using deltas by factor of learning rate
// W += alpha * (O x D)
// B += alpha * D
void layer_update(layer_t const *layer, double l_rate){
    for (int y = 0; y < layer->num_inputs; y++){
        for (int x = 0; x < layer->num_outputs; x++){
            layer->weights[y][x] += l_rate * layer->prev->outputs[y] * layer->deltas[x];
        }     
    }

    for (int i = 0; i < layer->num_outputs; i++){
        layer->biases[i] += l_rate * layer->deltas[i];
    }
}
