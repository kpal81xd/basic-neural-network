#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random.h"
#include "layer.h"

typedef struct ann {

    layer_t *input_layer;
    layer_t *output_layer;

} ann_t;

ann_t *ann_create(int num_layers, int *layer_outputs){
    ann_t *ann = malloc(sizeof(ann_t));
    layer_t *layers[num_layers];
    layers[0] = layer_create();
    layer_init(layers[0], layer_outputs[0], NULL);
    for (int i = 1; i < num_layers; i++){
        layers[i] = layer_create();
        layer_init(layers[i], layer_outputs[i], layers[i-1]);
    }
    ann->input_layer = layers[0];
    ann->output_layer = layers[num_layers - 1];
    
    return ann;
}

void ann_free(ann_t *ann){ 
    layer_t *current = ann->input_layer;
    layer_t *next;
    while(current){
        next = current->next;
        layer_free(current);
        current = next;
    }
    free(ann);
}

void ann_predict(ann_t const *ann, double const *inputs){
    layer_t *current = ann->input_layer;
    for (int i = 0; i < current->num_outputs; i++){
        current->outputs[i] = inputs[i];
    }
    current = current->next;
    while(current){
        layer_compute_outputs(current);
        current = current->next;
    }
}

void ann_train(ann_t *ann, double *inputs, double *targets, double l_rate){
    ann_predict(ann, inputs);
    layer_t *current = ann->output_layer;
    for (int i = 0; i < current->num_outputs; i++){
        current->deltas[i] = sigmoidPrime(current->outputs[i]) * (targets[i] - current->outputs[i]);
    }
    current = current->prev;
    while(current->prev){
        layer_compute_deltas(current);
        current = current->prev;
    }
    current = ann->output_layer;
    while(current->prev){
        layer_update(current, l_rate);
        current = current->prev;
    }
}

int main(void){
    
    srand(time(NULL));

    printf("Artificial Neural Network\n"); 
    printf("--------------------------\n");

    int numberOfLayers = 3;
    int layer_outputs[] = {2, 3, 1};
    ann_t *ann = ann_create(numberOfLayers, layer_outputs);
    printf("\nGenerated ANN\n");

    double inputs[] = {0, 1};
    ann_predict(ann, inputs);
    printf("\nUntrained Prediction\n");
    for (int i = 0; i < ann->output_layer->num_outputs; i++){
        printf("%lf ",ann->output_layer->outputs[i]);
    }
    printf("\n");

    double targets[] = {1};
    for (int i = 0; i < 1; i++){
        ann_train(ann, inputs, targets, 1);
    }
    printf("\nTrained\n");

    ann_predict(ann, inputs);
    printf("\nTrained Prediction\n");
    for (int i = 0; i < ann->output_layer->num_outputs; i++){
        printf("%lf ",ann->output_layer->outputs[i]);
    }
    printf("\n");

    ann_free(ann);
    printf("\nFreed ANN\n");

    return 0;
}