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
#define NUM_CLASSES 10  // Digits 0-9

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// ---------------------------------------
// Allocate contiguous memory for images/labels.
// Caller is responsible for freeing the returned pointer.
double* allocateContiguous(int numElements) {
    double* data = (double*)malloc(numElements * sizeof(double));
    if (!data) {
        fprintf(stderr, "Error allocating memory\n");
        exit(EXIT_FAILURE);
    }
    return data;
}

// ---------------------------------------
// Neural Network with contiguous weight arrays on host.
typedef struct {
    double* W1;  // Dimensions: HIDDEN_SIZE x INPUT_SIZE (flattened row-major)
    double* W2;  // Dimensions: OUTPUT_SIZE x HIDDEN_SIZE (flattened row-major)
    double* b1;  // Dimensions: HIDDEN_SIZE
    double* b2;  // Dimensions: OUTPUT_SIZE
} NeuralNetwork;

// Create a new network with weights and biases allocated contiguously.
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        net->W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        net->W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    return net;
}

// Free network memory.
void freeNetwork(NeuralNetwork* net) {
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

// ---------------------------------------
// Host (CPU) forward pass.
// input: pointer to one image (length INPUT_SIZE)
// hidden: pointer to hidden layer array (length HIDDEN_SIZE)
// output: pointer to output layer array (length OUTPUT_SIZE)
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Compute hidden layer: z = W1 * input + b1, then apply ReLU.
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = (sum > 0.0) ? sum : 0.0;
    }
    // Compute output layer: z = W2 * hidden + b2.
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }
    // Softmax activation for output.
    double max_x = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if(output[i] > max_x) max_x = output[i];
    }
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = exp(output[i] - max_x);
        sum += output[i];
    }
    if(sum == 0.0) {
        for (int i = 0; i < OUTPUT_SIZE; i++)
            output[i] = 1.0 / OUTPUT_SIZE;
    } else {
        for (int i = 0; i < OUTPUT_SIZE; i++)
            output[i] /= sum;
    }
}

// ---------------------------------------
// Device function: Softmax used in kernel.
__device__ void kernelsoftmax(double* x, int size) {
    double max_x = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_x)
            max_x = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_x);
        sum += x[i];
    }
    if(sum == 0.0) {
        for (int i = 0; i < size; i++)
            x[i] = 1.0 / size;
    } else {
        for (int i = 0; i < size; i++)
            x[i] /= sum;
    }
}

// ---------------------------------------
// CUDA Forward pass Kernel.
// d_W1: device pointer for W1 (HIDDEN_SIZE * INPUT_SIZE)
// d_W2: device pointer for W2 (OUTPUT_SIZE * HIDDEN_SIZE)
// d_b1: device pointer for b1 (HIDDEN_SIZE)
// d_b2: device pointer for b2 (OUTPUT_SIZE)
// input: pointer for one image (INPUT_SIZE)
// hidden: pointer for hidden activations (HIDDEN_SIZE)
// output: pointer for output activations (OUTPUT_SIZE)
__global__ void forward_kernel(double* d_W1, double* d_W2, double* d_b1, double* d_b2,
                               double* input, double* hidden, double* output) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    
    // Hidden layer: each thread computes one neuron if tid < HIDDEN_SIZE.
    if (tid < HIDDEN_SIZE) {
        double sum = d_b1[tid];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += d_W1[tid * INPUT_SIZE + j] * input[j];
        }
        hidden[tid] = (sum > 0.0) ? sum : 0.0;
    }
    __syncthreads();

    // Output layer: each thread computes one neuron if tid < OUTPUT_SIZE.
    if (tid < OUTPUT_SIZE) {
        double sum = d_b2[tid];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += d_W2[tid * HIDDEN_SIZE + j] * hidden[j];
        }
        output[tid] = sum;
    }
    __syncthreads();

    // One thread applies softmax.
    if (tid == 0) {
        kernelsoftmax(output, OUTPUT_SIZE);
    }
}

// ---------------------------------------
// CUDA Backward pass Kernel.
// d_W1, d_W2, d_b1, d_b2: device pointers for weights and biases.
// input: pointer for one image (INPUT_SIZE)
// hidden: pointer for hidden activations (HIDDEN_SIZE)
// output: pointer for output activations (OUTPUT_SIZE)
// target: pointer for one label vector (OUTPUT_SIZE)
__global__ void backward_kernel(double* d_W1, double* d_W2, double* d_b1, double* d_b2,
                                double* input, double* hidden, double* output, double* target) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Update hidden layer weights and biases in parallel.
    if (tid < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double d_output = output[j] - target[j];
            sum += d_W2[j * HIDDEN_SIZE + tid] * d_output;
        }
        double d_hidden = sum * ((hidden[tid] > 0.0) ? 1.0 : 0.0);
        for (int j = 0; j < INPUT_SIZE; j++) {
            d_W1[tid * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden * input[j];
        }
        d_b1[tid] -= LEARNING_RATE * d_hidden;
    }
    __syncthreads();

    // Use one thread to update the output layer.
    if (tid == 0) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            double d_output = output[i] - target[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                d_W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output * hidden[j];
            }
            d_b2[i] -= LEARNING_RATE * d_output;
        }
    }
}

// ---------------------------------------
// Load MNIST Images from file into a contiguous array.
// Returns a pointer to an array of doubles of length (numImages * INPUT_SIZE).
double* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 16, SEEK_SET);
    double* images = allocateContiguous(numImages * INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i * INPUT_SIZE + j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

// ---------------------------------------
// Load MNIST Labels from file into a contiguous array.
// Returns a pointer to an array of doubles of length (numLabels * OUTPUT_SIZE).
// Each label is one-hot encoded.
double* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 8, SEEK_SET);
    double* labels = allocateContiguous(numLabels * OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i * OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// ---------------------------------------
// Train network: run both CUDA kernel and serial forward passes for each image,
// compare the hidden and output activations, and perform backward pass.
void train(NeuralNetwork* net,
           double* d_W1, double* d_W2, double* d_b1, double* d_b2,
           double* H_images, double* H_labels, int numImages) {

    // Allocate device memory for entire training set input and label arrays.
    double* D_images;
    double* D_labels;
    cudaMalloc((void**)&D_images, numImages * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&D_labels, numImages * OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(D_images, H_images, numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(D_labels, H_labels, numImages * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Device memory for hidden and output layers (for one image at a time).
    double* D_hidden;
    double* D_output;
    cudaMalloc((void**)&D_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&D_output, OUTPUT_SIZE * sizeof(double));

    double h_output[OUTPUT_SIZE];

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // Compute pointer offsets for current image and label.
            double* d_input = D_images + (i * INPUT_SIZE);
            double* d_label = D_labels + (i * OUTPUT_SIZE);

            // --- GPU Forward Pass ---
            forward_kernel<<<4, 32>>>(d_W1, d_W2, d_b1, d_b2, d_input, D_hidden, D_output);
            cudaDeviceSynchronize();
  
            // --- Backward Pass (CUDA Kernel) ---
            int threadsPerBlock = 256;
            int blocks = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
            backward_kernel<<<blocks, threadsPerBlock>>>(d_W1, d_W2, d_b1, d_b2,
                                                          d_input, D_hidden, D_output, d_label);
            cudaDeviceSynchronize();

            // Copy the output from device to host for loss & accuracy computation.
            cudaMemcpy(h_output, D_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

            // Compute loss and accuracy.
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= H_labels[i * OUTPUT_SIZE + k] * log(h_output[k]);
            }
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (H_labels[i * OUTPUT_SIZE + j] > H_labels[i * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual)
                correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));

    cudaFree(D_images);
    cudaFree(D_labels);
    cudaFree(D_hidden);
    cudaFree(D_output);
}

// ---------------------------------------
// Evaluate network on test data using serial forward pass.
void evaluate(double* images, double* labels, int numImages,
              double* d_W1, double* d_W2, double* d_b1, double* d_b2) {

    int correct = 0;
    
    // Allocate device memory for hidden and output layers (for one image).
    double* D_hidden;
    double* D_output;
    cudaMalloc((void**)&D_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&D_output, OUTPUT_SIZE * sizeof(double));

    // For each test image, copy the image to a temporary device buffer,
    // launch the forward kernel, and then copy back the results.
    for (int i = 0; i < numImages; i++) {
        double* d_input;
        cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double));
        cudaMemcpy(d_input, images + (i * INPUT_SIZE),
                   INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        
        // Launch forward kernel.
        forward_kernel<<<4, 32>>>(d_W1, d_W2, d_b1, d_b2, d_input, D_hidden, D_output);
        cudaDeviceSynchronize();
        
        // Copy the output (activation of the output layer) back to host.
        double h_output[OUTPUT_SIZE];
        cudaMemcpy(h_output, D_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Determine the predicted label (index of max value in h_output).
        int pred = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred])
                pred = j;
        }
        // Extract the actual label from the contiguous labels array.
        int actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual])
                actual = j;
        }
        if (pred == actual)
            correct++;
        
        cudaFree(d_input);
    }
    
    printf("Test Accuracy (GPU Forward): %.2f%%\n", (correct / (double)numImages) * 100);
    
    cudaFree(D_hidden);
    cudaFree(D_output);
}


// ---------------------------------------
// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    // Load training and testing data as contiguous arrays.
    double* H_train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double* H_train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double* H_test_images  = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double* H_test_labels  = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    // Create network on host.
    NeuralNetwork* net = createNetwork();

    // Allocate device memory for weights and biases.
    double *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(double));

    // Copy host network weights and biases to device.
    cudaMemcpy(d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Train the network.
    train(net, d_W1, d_W2, d_b1, d_b2, H_train_images, H_train_labels, 60000);

    // Evaluate on test data.
    evaluate(H_test_images, H_test_labels, 10000, d_W1, d_W2, d_b1, d_b2);
    
    // Free host network and data.
    freeNetwork(net);
    free(H_train_images);
    free(H_train_labels);
    free(H_test_images);
    free(H_test_labels);

    // Free device weights and biases.
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_b1);
    cudaFree(d_b2);

    return 0;
}
