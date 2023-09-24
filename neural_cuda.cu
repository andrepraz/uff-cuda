#include <iostream>
#include <vector>
#include <openacc.h>

// Função de ativação sigmoid
__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Forward pass da rede neural em CUDA
__global__ void forwardPass(float* input, float* weights, float* output, int numInputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calcule a ativação da camada de saída
    float activation = 0.0;
    for (int i = 0; i < numInputs; ++i) {
        activation += input[i] * weights[i];
    }
    output[idx] = sigmoid(activation);
}

int main() {
    int numInputs = 10000;
    int numOutputs = 1;
    int numSamples = 1000;
    
    // Aloque memória na CPU para os dados e pesos
    float* inputData = new float[numInputs * numSamples];
    float* outputData = new float[numOutputs * numSamples];
    float* weights = new float[numInputs];
    
    // Inicialize os dados e pesos (suponha que você tenha seus dados aqui)
    
    // Aloque memória na GPU
    float* d_inputData;
    float* d_outputData;
    float* d_weights;
    
    cudaMalloc(&d_inputData, numInputs * numSamples * sizeof(float));
    cudaMalloc(&d_outputData, numOutputs * numSamples * sizeof(float));
    cudaMalloc(&d_weights, numInputs * sizeof(float));
    
    // Copie dados e pesos da CPU para a GPU
    cudaMemcpy(d_inputData, inputData, numInputs * numSamples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, numInputs * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configurar grade CUDA
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;
    
    // Executar a forward pass em CUDA
    #pragma acc parallel loop present(d_inputData, d_weights, d_outputData)
    for (int i = 0; i < numSamples; ++i) {
        forwardPass<<<gridSize, blockSize>>>(d_inputData + i * numInputs, d_weights, d_outputData + i, numInputs);
    }
    
    // Copie os resultados de volta da GPU para a CPU
    cudaMemcpy(outputData, d_outputData, numOutputs * numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Libere memória na GPU
    cudaFree(d_inputData);
    cudaFree(d_outputData);
    cudaFree(d_weights);
    
    // Libere memória na CPU
    delete[] inputData;
    delete[] outputData;
    delete[] weights;
    
    return 0;
}