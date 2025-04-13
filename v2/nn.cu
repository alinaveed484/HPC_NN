#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h> 

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

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
__device__ void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

__device__ void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Function to offload a NeuralNetwork from host to device
NeuralNetwork* allocateDeviceNetwork(NeuralNetwork* h_net) {
    double *d_W1, *d_W2, *d_b1, *d_b2;

    // Allocate device memory for flat weight matrices and biases.
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(double));

    // Copy host network weights and biases to device.
    cudaMemcpy(d_W1, h_net->W1[0], HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_net->W2[0], OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Build a host-side NeuralNetwork struct that holds the device pointers.
    // Note: We reinterpret d_W1 and d_W2 as double** for now.
    NeuralNetwork h_net_dev;
    h_net_dev.W1 = (double**)d_W1;  // Actually, on device this will be used as a flat array.
    h_net_dev.W2 = (double**)d_W2;  // Ensure your device kernels treat these as flat arrays.
    h_net_dev.b1 = d_b1;
    h_net_dev.b2 = d_b2;

    // Allocate device memory for the NeuralNetwork struct.
    NeuralNetwork* d_net;
    cudaMalloc((void**)&d_net, sizeof(NeuralNetwork));
    // Copy our constructed host struct (with device pointers) to device memory.
    cudaMemcpy(d_net, &h_net_dev, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);

    return d_net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

// Forward pass CUDA FUNCTION
__global__ void forward_kernel(NeuralNetwork* net, double* input, double* hidden, double* output) {
    int tid = threadIdx.x + (blockIdx.x * blockdim.x);
    
    // Compute hidden layer activations in parallel (for tid < HIDDEN_SIZE).
    if(tid < HIDDEN_SIZE) {
        double sum = net->b1[tid];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[tid * INPUT_SIZE + j] * input[j];
        }
        // Apply ReLU activation.
        hidden[tid] = (sum > 0.0) ? sum : 0.0;
    }
    __syncthreads(); // Ensure all hidden neurons are computed.
    
    // Compute output layer activations in parallel (for tid < OUTPUT_SIZE).
    if(tid < OUTPUT_SIZE) {
        double sum = net->b2[tid];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[tid * HIDDEN_SIZE + j] * hidden[j];
        }
        output[tid] = sum;
    }
    __syncthreads();
    
    // Let one thread perform the softmax (or consider a parallel softmax if needed).
    if(tid == 0) {
        softmax(output, OUTPUT_SIZE);
    }
}
// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update weights (gradient descent)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}

// Train network
void train(NeuralNetwork* net, double** H_images, double** H_labels, int numImages) {

    // Device pointers
    double* D_images;
    double* D_labels;
    double* D_hidden;
    double* D_output;

    // Allocate memory on device
    cudaMalloc((void**)&D_images, numImages * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&D_labels, numImages * OUTPUT_SIZE * sizeof(double));

    //copying data to the device side.
    cudaMemcpy(D_images, H_images[0], numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(D_labels, H_labels[0], numImages * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);


    cudaMalloc(&D_hidden, numImages * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&D_output, numImages * OUTPUT_SIZE * sizeof(double));


    double h_output[OUTPUT_SIZE];

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
           // double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            double* hidden = &D_hidden[i * HIDDEN_SIZE];
            double* output = &D_output[i * OUTPUT_SIZE];
            
            forward_kernel<<<4,32>>>(net, D_images+(i * INPUT_SIZE), hidden, output);

            cudaDeviceSynchronize();

            backward(net, D_images+(i * INPUT_SIZE), hidden, output, D_labels + (i * OUTPUT_SIZE));

            cudaDeviceSynchronize();

            // Copy the output for the current image from device to host
            cudaMemcpy(h_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(h_output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}


double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}
void freeDeviceNetwork(NeuralNetwork* d_net) {
    NeuralNetwork h_net;
    // Copy the structure from device to host
    cudaMemcpy(&h_net, d_net, sizeof(NeuralNetwork), cudaMemcpyDeviceToHost);
    
    // Free individual device allocations using the pointers from h_net
    cudaFree(h_net.W1); // This frees d_W1
    cudaFree(h_net.W2); // This frees d_W2
    cudaFree(h_net.b1); // This frees d_b1
    cudaFree(h_net.b2); // This frees d_b2

    // Finally, free the device NeuralNetwork struct
    cudaFree(d_net);
}

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** H_train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** H_train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** H_test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** H_test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);



    NeuralNetwork* net = createNetwork();
    NeuralNetwork* d_net = allocateDeviceNetwork(net);  //get the network on device.


    train(d_net, H_train_images, H_train_labels, 60000);

    evaluate(d_net, H_test_images, H_test_labels, 10000);

    freeNetwork(net);
    freeDeviceNetwork(d_net);
    return 0;
}
