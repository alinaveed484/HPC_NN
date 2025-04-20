/* Revised MNIST Neural Network with OpenACC */
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

// Timer
static double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate 1D matrix
static double* allocateMatrix1D(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

// Activation functions
static void relu(double* x, int size) {
    #pragma acc parallel loop present(x[0:size])
    for (int i = 0; i < size; i++) x[i] = (x[i] > 0 ? x[i] : 0);
}
static void softmax(double* x, int size) {
    double sum = 0;
    #pragma acc parallel loop present(x[0:size]) reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]); sum += x[i];
    }
    #pragma acc parallel loop present(x[0:size])
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// Neural network struct
typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize network (host-only)
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix1D(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix1D(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(777);
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->W2[i] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Forward pass (device)
void forward(NeuralNetwork* net, double* in, double* hidden, double* out) {
    #pragma acc parallel loop present(net, net->W1[0:HIDDEN_SIZE*INPUT_SIZE], net->b1[0:HIDDEN_SIZE], in[0:INPUT_SIZE], hidden[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) sum += net->W1[i * INPUT_SIZE + j] * in[j];
       // hidden[i] = sum;
        hidden[i] = (sum > 0 ? sum : 0); // ReLU merged here
    }
    relu(hidden, HIDDEN_SIZE);

    #pragma acc parallel loop present(net, net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE], hidden[0:HIDDEN_SIZE], out[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
        out[i] = sum;
    }
    softmax(out, OUTPUT_SIZE);
}
// Backpropagation (device)
void backward(NeuralNetwork* net, double* in, double* hidden, double* out, double* tgt, double* d_out, double* d_hid) {
    int i, j;

    // Output gradient (small, so no pragma)
    #pragma acc parallel present(out[0:OUTPUT_SIZE], tgt[0:OUTPUT_SIZE], d_out[0:OUTPUT_SIZE])
    for (i = 0; i < OUTPUT_SIZE; i++) {
        d_out[i] = out[i] - tgt[i];
    }

    // Hidden gradient
    #pragma acc parallel loop present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], d_out[0:OUTPUT_SIZE], hidden[0:HIDDEN_SIZE], d_hid[0:HIDDEN_SIZE])
    for (i = 0; i < HIDDEN_SIZE; i++) {
        double sum_grad = 0.0;
        for (j = 0; j < OUTPUT_SIZE; j++)
            sum_grad += net->W2[j * HIDDEN_SIZE + i] * d_out[j];
        d_hid[i] = sum_grad * (hidden[i] > 0 ? 1.0 : 0.0);
    }

    // Update W2
    #pragma acc parallel loop collapse(2) present(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], d_out[0:OUTPUT_SIZE], hidden[0:HIDDEN_SIZE])
    for (i = 0; i < OUTPUT_SIZE; i++)
        for (j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_out[i] * hidden[j];

    // Update W1
    #pragma acc parallel loop collapse(2) present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], d_hid[0:HIDDEN_SIZE], in[0:INPUT_SIZE])
    for (i = 0; i < HIDDEN_SIZE; i++)
        for (j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hid[i] * in[j];

    // Update biases (small loops, so no loop pragma)
    //#pragma acc parallel present(net->b2[0:OUTPUT_SIZE], d_out[0:OUTPUT_SIZE])
    for (i = 0; i < OUTPUT_SIZE; i++) net->b2[i] -= LEARNING_RATE * d_out[i];

    #pragma acc parallel present(net->b1[0:HIDDEN_SIZE], d_hid[0:HIDDEN_SIZE])
    for (i = 0; i < HIDDEN_SIZE; i++) net->b1[i] -= LEARNING_RATE * d_hid[i];
}
// Train (host delegating to device)

void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t t0 = clock();

    #pragma acc data copyin(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE]) \
                     copy(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])
    {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE], d_out[OUTPUT_SIZE], d_hid[HIDDEN_SIZE];
        #pragma acc enter data create(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE], d_out[0:OUTPUT_SIZE], d_hid[0:HIDDEN_SIZE])

        for (int ep = 0; ep < EPOCHS; ep++) {
            clock_t e0 = clock();
            double loss = 0; int correct = 0;
            double fwd_time = 0.0;
            double bwd_time = 0.0;

            for (int i_img = 0; i_img < numImages; i_img++) {
                double* img = &images[i_img * INPUT_SIZE];
                double* lbl = &labels[i_img * OUTPUT_SIZE];

                clock_t fwd_start = clock();
                forward(net, img, hidden, output);
                #pragma acc update self(output[0:OUTPUT_SIZE])
                clock_t fwd_end = clock();
                fwd_time += (double)(fwd_end - fwd_start) / CLOCKS_PER_SEC;

                for (int k = 0; k < OUTPUT_SIZE; k++) loss -= lbl[k] * log(output[k]);
                int pred = 0, act = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (output[j] > output[pred]) pred = j;
                    if (lbl[j]    > lbl[act])    act  = j;
                }
                if (pred == act) correct++;

                clock_t bwd_start = clock();
                backward(net, img, hidden, output, lbl, d_out, d_hid);
                clock_t bwd_end = clock();
                bwd_time += (double)(bwd_end - bwd_start) / CLOCKS_PER_SEC;
            }
            printf("Epoch %d - Loss: %.4f - Acc: %.2f%% - FWD: %.3fs - BWD: %.3fs - Time: %.3fs\n", 
                   ep+1, loss/numImages, 100.0*correct/numImages, fwd_time, bwd_time, get_time(e0));
        }

        #pragma acc exit data delete(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE], d_out[0:OUTPUT_SIZE], d_hid[0:HIDDEN_SIZE])
    }
    printf("Total training time: %.3fs\n", get_time(t0));
}

// Evaluate (host delegating to device)
void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    int correct = 0;
    #pragma acc data copyin(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE]) \
                     copyin(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])
    {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        #pragma acc enter data create(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])

        for (int i_img = 0; i_img < numImages; i_img++) {
            double* img = &images[i_img * INPUT_SIZE];
            double* lbl = &labels[i_img * OUTPUT_SIZE];
            forward(net, img, hidden, output);
            #pragma acc update self(output[0:OUTPUT_SIZE])

            int pred = 0, act = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (lbl[j]    > lbl[act])    act  = j;
            }
            if (pred == act) correct++;
        }

        #pragma acc exit data delete(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
    }
    printf("Test Accuracy: %.2f%%\n", 100.0 * correct / numImages);
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

int main() {
    printf("MNIST Neural Network (OpenACC)\n\n");
    double* train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double* train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double* test_images  = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double* test_labels  = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    free(train_images); free(train_labels);
    free(test_images);  free(test_labels);
    return 0;
}
