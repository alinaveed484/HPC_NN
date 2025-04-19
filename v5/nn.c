#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a 1D matrix
double* allocateMatrix1D(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

// Activation functions
void relu(double* x, int size) {
    #pragma acc parallel loop present(x)
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    #pragma acc parallel loop present(x) reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    #pragma acc parallel loop present(x)
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix1D(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix1D(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(777);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    // Explicitly copy weights to device
    #pragma acc enter data copyin(net[0:1], net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
        net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])
    return net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Compute hidden layer
    #pragma acc parallel loop present(net, net->W1, net->b1, input, hidden)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = sum;
    }
    relu(hidden, HIDDEN_SIZE);

    // Compute output layer
    #pragma acc parallel loop present(net, net->W2, net->b2, hidden, output)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }
    softmax(output, OUTPUT_SIZE);
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    #pragma acc data create(d_output, d_hidden)
    {
        // Compute output layer gradient
        //#pragma acc parallel loop present(output, target)
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            d_output[i] = output[i] - target[i];
        }

        // Compute hidden layer gradient
        #pragma acc parallel loop present(net->W2, hidden, d_output)
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                sum += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
            }
            d_hidden[i] = sum * (hidden[i] > 0 ? 1.0 : 0.0);
        }

        // Update weights (gradient descent)
        // Update W2
        #pragma acc parallel loop collapse(2) present(net->W2, hidden, d_output)
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
            }
        }

        // Update W1
        #pragma acc parallel loop collapse(2) present(net->W1, input, d_hidden)
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
            }
        }

        // Update b2
        //#pragma acc parallel loop present(net->b2, d_output)
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            net->b2[i] -= LEARNING_RATE * d_output[i];
        }

        // Update b1
        //#pragma acc parallel loop present(net->b1, d_hidden)
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            net->b1[i] -= LEARNING_RATE * d_hidden[i];
        }
    }
}

// Train network
void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t total_start = clock();
    #pragma acc enter data copyin(net[0:1], net->W1[0:HIDDEN_SIZE*INPUT_SIZE], net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE], images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        double forward_time = 0.0, backward_time = 0.0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            double* current_image = &images[i * INPUT_SIZE];
            double* current_label = &labels[i * OUTPUT_SIZE];

            #pragma acc enter data create(hidden, output)
            clock_t f_start = clock();
            forward(net, current_image, hidden, output);
            forward_time += get_time(f_start);

            clock_t b_start = clock();
            backward(net, current_image, hidden, output, current_label);
            backward_time += get_time(b_start);

            #pragma acc update self(output[0:OUTPUT_SIZE], current_label[0:OUTPUT_SIZE])
            #pragma acc exit data delete(hidden, output)

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) 
                loss -= current_label[k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (current_label[j] > current_label[actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs - (FWD: %.3fs, BWD: %.3fs)\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start), forward_time, backward_time);
    }
    #pragma acc exit data copyout(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])
    #pragma acc exit data delete(images, labels, net)
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    int correct = 0;
    #pragma acc enter data copyin(net[0:1], net->W1, net->W2, net->b1, net->b2, images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        double* current_image = &images[i * INPUT_SIZE];
        #pragma acc enter data create(hidden, output)
        forward(net, current_image, hidden, output);
        #pragma acc update self(output[0:OUTPUT_SIZE], labels[i*OUTPUT_SIZE:OUTPUT_SIZE])
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i*OUTPUT_SIZE + j] > labels[i*OUTPUT_SIZE + actual]) actual = j;
        }
        if (pred == actual) correct++;
        #pragma acc exit data delete(hidden, output)
    }
    #pragma acc exit data delete(images, labels, net)
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double* images = allocateMatrix1D(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, file);
            images[i * INPUT_SIZE + j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double* labels = allocateMatrix1D(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i * OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free allocated memory
void freeNetwork(NeuralNetwork* net) {
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double* train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double* train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double* test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double* test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    return 0;
}