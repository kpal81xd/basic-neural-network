#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random.h"
#include "ann.h"
#include "layer.h"

#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1

#define BATCH_SIZE 4

#define EPOCHS 1000000

// Create the neural network
ann_t *ann_create(const int num_layers, int *layer_outputs){
    ann_t *ann = malloc(sizeof(ann_t));
    if (!ann) return NULL;
    layer_t *layers[num_layers];
    layers[0] = layer_create();
    if (!layers[0]) return NULL;
    if (layer_init(layers[0], layer_outputs[0], NULL)) return NULL;
    for (int i = 1; i < num_layers; i++){
        layers[i] = layer_create();
        if (!layers[i]) return NULL;
        if (layer_init(layers[i], layer_outputs[i], layers[i-1])) return NULL;
    }
    ann->input_layer = layers[0];
    ann->output_layer = layers[num_layers - 1];
    
    return ann;
}

// Free neural network
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

// Train the neural network using the sigmoid activation function
// Use back propagation to update the weights of the hidden layers
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

// Predict results using trained neural network
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

// Neural Network to train and learn XOR
int main(void){
    
    srand(time(NULL));

    printf("Artificial Neural Network\n"); 
    printf("--------------------------\n");

    // Define constants
    int numberOfLayers = 3;
    int layer_outputs[] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};

    // Define input and target data
    double inputs[BATCH_SIZE][INPUT_SIZE] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double targets[BATCH_SIZE][OUTPUT_SIZE] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Create neural network
    ann_t *ann = ann_create(numberOfLayers, layer_outputs);
    if (!ann) {
      printf("\nANN Memory Allocation Failed\n");
      return 1;
    }
    printf("\nGenerated ANN\n");

    // Predict outputs with untrained neural network
    printf("\nUntrained Prediction\n");
    for(int j = 0; j < BATCH_SIZE; j++){
        ann_predict(ann, inputs[j]);
        printf("[");
        for(int i = 0; i < INPUT_SIZE; i++){
            printf("%d", (int) inputs[j][i]);
            if (i < INPUT_SIZE-1){
                printf(",");
            }
        }
        printf("] - ");
        for (int i = 0; i < ann->output_layer->num_outputs; i++){
            printf("%lf ",ann->output_layer->outputs[i]);
        }
        printf("\n");
    }

    // Train neural network with learning rate 1
    // (May need to adjust for overfitting)
    for (int i = 0; i < EPOCHS; i++){
        for (int j = 0; j < BATCH_SIZE; j++){
            ann_train(ann, inputs[j], targets[j], 1);
        }
    }
    printf("\nTraining Complete - %d Epochs\n", EPOCHS);

    // Predict outputs with trained neural network
    printf("\nTrained Prediction\n");
    for(int j = 0; j < BATCH_SIZE; j++){
        ann_predict(ann, inputs[j]);
        printf("[");
        for(int i = 0; i < INPUT_SIZE; i++){
            printf("%d", (int) inputs[j][i]);
            if (i < INPUT_SIZE-1){
                printf(",");
            }
        }
        printf("] - ");
        for (int i = 0; i < ann->output_layer->num_outputs; i++){
            printf("%lf ",ann->output_layer->outputs[i]);
        }
        printf("\n");
    }

    // Display expected outputs
    printf("\nExpected\n");
    for(int j = 0; j < BATCH_SIZE; j++){
        ann_predict(ann, inputs[j]);
        printf("[");
        for(int i = 0; i < INPUT_SIZE; i++){
            printf("%d", (int) inputs[j][i]);
            if (i < INPUT_SIZE-1){
                printf(",");
            }
        }
        printf("] - ");
        for (int i = 0; i < OUTPUT_SIZE; i++){
            printf("%lf ",targets[j][i]);
        }
        printf("\n");
    }

    // Free neural network
    ann_free(ann);
    printf("\nFreed ANN\n");

    return 0;
}