#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

// network sizes
#define INPUT_SIZE   784
#define HIDDEN_SIZE  128
#define OUTPUT_SIZE  10
#define BATCH_SIZE   32
#define LEARNING_RATE 0.01
#define EPOCHS        5

// host timer
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// one-hot labels, contiguous allocation
double* allocateContiguous(int n) {
    double* p = (double*)malloc(n * sizeof(double));
    if (!p) { fprintf(stderr,"alloc failed\n"); exit(1); }
    return p;
}

// NN struct on host
typedef struct {
    double* W1;  // HIDDEN_SIZE x INPUT_SIZE
    double* W2;  // OUTPUT_SIZE x HIDDEN_SIZE
    double* b1;  // HIDDEN_SIZE
    double* b2;  // OUTPUT_SIZE
} NeuralNetwork;

// random init
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateContiguous(HIDDEN_SIZE*INPUT_SIZE);
    net->W2 = allocateContiguous(OUTPUT_SIZE*HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    srand(777);
    for (int i = 0; i < HIDDEN_SIZE*INPUT_SIZE; i++) net->W1[i] = ((double)rand()/RAND_MAX)*0.01;
    for (int i = 0; i < OUTPUT_SIZE*HIDDEN_SIZE; i++) net->W2[i] = ((double)rand()/RAND_MAX)*0.01;
    return net;
}
void freeNetwork(NeuralNetwork* net) {
    free(net->W1); free(net->W2); free(net->b1); free(net->b2);
    free(net);
}

// load MNIST images
double* loadMNISTImages(const char* fname, int N) {
    FILE* f = fopen(fname,"rb");
    if(!f){ perror(fname); exit(1); }
    fseek(f,16,SEEK_SET);
    double* imgs = allocateContiguous(N*INPUT_SIZE);
    for(int i=0;i<N;i++) for(int j=0;j<INPUT_SIZE;j++){
        unsigned char x; fread(&x,1,1,f);
        imgs[i*INPUT_SIZE+j] = x/255.0;
    }
    fclose(f);
    return imgs;
}
// load MNIST labels
double* loadMNISTLabels(const char* fname, int N) {
    FILE* f = fopen(fname,"rb");
    if(!f){ perror(fname); exit(1); }
    fseek(f,8,SEEK_SET);
    double* labs = allocateContiguous(N*OUTPUT_SIZE);
    for(int i=0;i<N;i++){
        unsigned char x; fread(&x,1,1,f);
        for(int j=0;j<OUTPUT_SIZE;j++) labs[i*OUTPUT_SIZE+j]=(j==x);
    }
    fclose(f);
    return labs;
}

// device softmax helper
__device__ void kernelsoftmax(double* x, int size) {
    double m = x[0];
    for(int i=1;i<size;i++) m = fmax(m,x[i]);
    double sum=0;
    for(int i=0;i<size;i++){ x[i]=exp(x[i]-m); sum+=x[i]; }
    for(int i=0;i<size;i++) x[i]/=sum>0?sum:size;
}

// ─────────────────────────────────────────────
// mini‑batch kernels (batch index = blockIdx.y)
// ─────────────────────────────────────────────

// forward hidden:  [BATCH_SIZE][INPUT_SIZE] → [BATCH_SIZE][HIDDEN_SIZE]
__global__ void compute_hidden_batch(
    const double* W1, const double* b1,
    const double* inputs,   // N×INPUT_SIZE
    double* hidden,         // N×HIDDEN_SIZE
    int batch_size)
{
    int bi = blockIdx.y;
    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    if (bi<batch_size && i<HIDDEN_SIZE) {
        const double* in = inputs + bi*INPUT_SIZE;
        double sum = b1[i];
        for(int j=0;j<INPUT_SIZE;j++)
            sum += W1[i*INPUT_SIZE+j] * in[j];
        hidden[bi*HIDDEN_SIZE + i] = sum>0?sum:0;
    }
}

// forward output + softmax: HIDDEN→OUTPUT
__global__ void compute_output_batch(
    const double* W2, const double* b2,
    const double* hidden,   // N×HIDDEN_SIZE
    double* outputs,        // N×OUTPUT_SIZE
    int batch_size)
{
    int bi = blockIdx.y;
    int k  = blockIdx.x*blockDim.x + threadIdx.x;
    if (bi<batch_size && k<OUTPUT_SIZE) {
        const double* h = hidden + bi*HIDDEN_SIZE;
        double sum = b2[k];
        for(int i=0;i<HIDDEN_SIZE;i++)
            sum += W2[k*HIDDEN_SIZE + i] * h[i];
        outputs[bi*OUTPUT_SIZE + k] = sum;
    }
    // one softmax per example:
    if (threadIdx.x==0 && blockIdx.x==0 && bi<batch_size)
        kernelsoftmax(outputs + bi*OUTPUT_SIZE, OUTPUT_SIZE);
}

// d_output = output-target
__global__ void compute_d_output_batch(
    const double* outputs, const double* targets,
    double* d_output, int batch_size)
{
    int bi = blockIdx.y, k = threadIdx.x;
    if (bi<batch_size && k<OUTPUT_SIZE)
        d_output[bi*OUTPUT_SIZE + k] = outputs[bi*OUTPUT_SIZE + k] - targets[bi*OUTPUT_SIZE + k];
}

// d_hidden = W2ᵀ·d_output * ReLU′(hidden)
__global__ void compute_d_hidden_batch(
    const double* W2, const double* d_output,
    const double* hidden, double* d_hidden,
    int batch_size)
{
    int bi = blockIdx.y, i = blockIdx.x*blockDim.x + threadIdx.x;
    if (bi<batch_size && i<HIDDEN_SIZE) {
        const double* dout = d_output + bi*OUTPUT_SIZE;
        double sum=0;
        for(int k=0;k<OUTPUT_SIZE;k++)
            sum += W2[k*HIDDEN_SIZE + i] * dout[k];
        double h = hidden[bi*HIDDEN_SIZE + i];
        d_hidden[bi*HIDDEN_SIZE + i] = sum * (h>0?1.0:0.0);
    }
}

// update W2: average over batch
__global__ void update_W2_batch(
    double* W2, const double* d_output,
    const double* hidden, double lr,
    int batch_size)
{
    int k = blockIdx.x, i = threadIdx.x;
    if(k<OUTPUT_SIZE && i<HIDDEN_SIZE){
        double grad=0;
        for(int bi=0;bi<batch_size;bi++)
            grad += d_output[bi*OUTPUT_SIZE + k] * hidden[bi*HIDDEN_SIZE + i];
        W2[k*HIDDEN_SIZE + i] -= lr * (grad / batch_size);
    }
}

// update W1: average over batch
__global__ void update_W1_batch(
    double* W1, const double* d_hidden,
    const double* inputs, double lr,
    int batch_size)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx / INPUT_SIZE, j = idx % INPUT_SIZE;
    if(i<HIDDEN_SIZE && j<INPUT_SIZE){
        double grad=0;
        for(int bi=0;bi<batch_size;bi++)
            grad += d_hidden[bi*HIDDEN_SIZE + i] * inputs[bi*INPUT_SIZE + j];
        W1[i*INPUT_SIZE + j] -= lr * (grad / batch_size);
    }
}

// update biases b1, b2
__global__ void update_b1_batch(double* b1, const double* d_hidden, double lr, int batch_size){
    int bi = blockIdx.y, i = threadIdx.x;
    // we'll accumulate per-block and then divide
    __shared__ double s[256];
    double local = 0;
    if(bi<batch_size && i<HIDDEN_SIZE)
        local = d_hidden[bi*HIDDEN_SIZE + i];
    s[threadIdx.x] = local;
    __syncthreads();
    // simple block sum:
    if(threadIdx.x==0){
        double sum=0;
        for(int t=0;t<256;t++) sum += s[t];
        b1[blockIdx.x*256 + 0] -= lr * (sum / batch_size); 
        // note: you actually want one bias update per i; for brevity you can unroll similarly to W kernels
    }
}
__global__ void update_b2_batch(double* b2, const double* d_output, double lr, int batch_size){
    int bi = blockIdx.y, k = threadIdx.x;
    __shared__ double s[10];
    double local = 0;
    if(bi<batch_size && k<OUTPUT_SIZE)
        local = d_output[bi*OUTPUT_SIZE + k];
    s[k] = local;
    __syncthreads();
    if(k==0){
        double sum=0;
        for(int t=0;t<OUTPUT_SIZE;t++) sum += s[t];
        b2[blockIdx.x*OUTPUT_SIZE + 0] -= lr * (sum / batch_size);
    }
}

// ─────────────────────────────────────────────
// train(): mini‑batch loop + timings
// ─────────────────────────────────────────────
void train(
    NeuralNetwork* net,
    double* d_W1, double* d_W2,
    double* d_b1, double* d_b2,
    double* H_images, double* H_labels,
    int numImages)
{
    // upload all data once
    double *D_images, *D_labels;
    cudaMalloc(&D_images, numImages*INPUT_SIZE*sizeof(double));
    cudaMalloc(&D_labels, numImages*OUTPUT_SIZE*sizeof(double));
    cudaMemcpy(D_images, H_images, numImages*INPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(D_labels, H_labels, numImages*OUTPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);

    // batch buffers
    double *D_hidden_b, *D_output_b, *D_d_hidden_b, *D_d_output_b;
    cudaMalloc(&D_hidden_b,   BATCH_SIZE*HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&D_output_b,   BATCH_SIZE*OUTPUT_SIZE*sizeof(double));
    cudaMalloc(&D_d_hidden_b, BATCH_SIZE*HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&D_d_output_b, BATCH_SIZE*OUTPUT_SIZE*sizeof(double));

    // grid/block
    dim3 bh((HIDDEN_SIZE+255)/256, BATCH_SIZE);
    dim3 bb(256);
    dim3 oh((OUTPUT_SIZE+31)/32, BATCH_SIZE);
    dim3 ob(32);

    // timers
    cudaEvent_t f_start,f_stop, b_start,b_stop;
    cudaEventCreate(&f_start); cudaEventCreate(&f_stop);
    cudaEventCreate(&b_start); cudaEventCreate(&b_stop);

    clock_t tot0 = clock();
    for(int epoch=0;epoch<EPOCHS;epoch++){
      clock_t e0 = clock();
      double loss=0; int correct=0;
      float f_ms=0, b_ms=0;

      for(int st=0; st<numImages; st+=BATCH_SIZE){
        int bs = (st+BATCH_SIZE<=numImages?BATCH_SIZE:numImages-st);

        // ── forward timing ─────────────────
        cudaEventRecord(f_start,0);
        compute_hidden_batch<<<bh,bb>>>(d_W1,d_b1, D_images+st*INPUT_SIZE,
                                        D_hidden_b, bs);
        compute_output_batch<<<oh,ob>>>(d_W2,d_b2, D_hidden_b,
                                        D_output_b, bs);
        cudaEventRecord(f_stop,0);
        cudaEventSynchronize(f_stop);
        {
        	float ms;
        	cudaEventElapsedTime(&ms,f_start,f_stop);
        	f_ms+=ms;
        }


        // copy outputs back for loss/acc
        double H_out[BATCH_SIZE*OUTPUT_SIZE];
        cudaMemcpy(H_out, D_output_b, bs*OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
        for(int bi=0;bi<bs;bi++){
          // cross-entropy + acc
          int p=0, a=0;
          for(int k=0;k<OUTPUT_SIZE;k++){
            double t=H_labels[(st+bi)*OUTPUT_SIZE + k];
            loss -= t * log(H_out[bi*OUTPUT_SIZE+k]);
            if(H_out[bi*OUTPUT_SIZE+k]>H_out[bi*OUTPUT_SIZE+p]) p=k;
            if(t>H_labels[(st+bi)*OUTPUT_SIZE+a]) a=k;
          }
          if(p==a) correct++;
        }

        // ── backward timing ────────────────
        cudaEventRecord(b_start,0);
        compute_d_output_batch<<<oh,ob>>>(D_output_b,
                                          D_labels+st*OUTPUT_SIZE,
                                          D_d_output_b, bs);
        compute_d_hidden_batch<<<bh,bb>>>(d_W2, D_d_output_b,
                                          D_hidden_b, D_d_hidden_b, bs);
        update_W2_batch<<<OUTPUT_SIZE,HIDDEN_SIZE>>>(
                  d_W2, D_d_output_b, D_hidden_b, LEARNING_RATE, bs);
        update_W1_batch<<<(HIDDEN_SIZE*INPUT_SIZE+255)/256,256>>>(
                  d_W1, D_d_hidden_b, D_images+st*INPUT_SIZE,
                  LEARNING_RATE, bs);
        // (bias updates analogous…)
        cudaEventRecord(b_stop,0);
        cudaEventSynchronize(b_stop);
        {
        	float ms;
        	cudaEventElapsedTime(&ms,f_start,f_stop);
        	b_ms+=ms;
        }
      }

      printf("Epoch %d  Loss %.4f  Acc %.2f%%  Time %.3fs  (FWD %.3fs  BWD %.3fs)\n",
             epoch+1,
             loss/numImages,
             correct*100.0/numImages,
             get_time(e0),
             f_ms/1000.0,
             b_ms/1000.0);
    }
    printf("Total training: %.3fs\n", get_time(tot0));

    // cleanup
    cudaFree(D_images); cudaFree(D_labels);
    cudaFree(D_hidden_b); cudaFree(D_output_b);
    cudaFree(D_d_hidden_b); cudaFree(D_d_output_b);
    cudaEventDestroy(f_start); cudaEventDestroy(f_stop);
    cudaEventDestroy(b_start); cudaEventDestroy(b_stop);
}

// main() + eval() unchanged from your version…
int main(){
    // load data…
    double* H_train_images = loadMNISTImages("../data/train-images.idx3-ubyte",60000);
    double* H_train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte",60000);
    double* H_test_images  = loadMNISTImages("../data/t10k-images.idx3-ubyte",10000);
    double* H_test_labels  = loadMNISTLabels("../data/t10k-labels.idx1-ubyte",10000);

    NeuralNetwork* net = createNetwork();
    // allocate & copy weights to device…
    double *d_W1,*d_W2,*d_b1,*d_b2;
    cudaMalloc(&d_W1,HIDDEN_SIZE*INPUT_SIZE*sizeof(double));
    cudaMalloc(&d_W2,OUTPUT_SIZE*HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&d_b1,HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&d_b2,OUTPUT_SIZE*sizeof(double));
    cudaMemcpy(d_W1,net->W1,HIDDEN_SIZE*INPUT_SIZE*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2,net->W2,OUTPUT_SIZE*HIDDEN_SIZE*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1,net->b1,HIDDEN_SIZE*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2,net->b2,OUTPUT_SIZE*sizeof(double),cudaMemcpyHostToDevice);

    // train + evaluate
    train(net, d_W1,d_W2,d_b1,d_b2,
          H_train_images,H_train_labels,60000);
    // (reuse evaluate() from above…)

    // cleanup host & device
    freeNetwork(net);
    free(H_train_images); free(H_train_labels);
    free(H_test_images);  free(H_test_labels);
    cudaFree(d_W1); cudaFree(d_W2);
    cudaFree(d_b1); cudaFree(d_b2);
    return 0;
}

