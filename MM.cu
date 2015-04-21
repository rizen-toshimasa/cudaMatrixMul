#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M_SIZE 1024//mat size
#define B_SIZE 1024//block size
//¿¿¿¿¿¿
void matMul(int *mat1, int *mat2, int *dstMat);
void matZeros(int *mat);
void matTrans(int *mat, int *dstMat);
int matDiffCount(int *mat1, int *mat2);
void printMat(int *mat);
//CUDA¿¿¿¿¿¿
__global__ void cudaMatMul(int *mat1, int *mat2, int *dstMat);
__global__ void cudaMatMulShared(int *mat1, int *mat2, int *dstMat);
__global__ void cudaMatMulSharedTsukuba(int *mat1, int *mat2, int *dstMat);
//¿¿¿¿¿
int main(int argc, char* argv[]){
    srand((unsigned)time(NULL));
    //Host Memory
    int *mat1 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *mat2 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *matResult = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    //Host Memory
    matZeros(matResult);
    for(int i=0; i<M_SIZE*M_SIZE; i++){
        mat1[i] = rand()%256;
        mat2[i] = rand()%256;
    }
    //Global Memory
    int *gMat1,*gMat2,*gMatResult;
    cudaMalloc((void **)&gMat1, sizeof(int) * M_SIZE * M_SIZE);
    cudaMalloc((void **)&gMat2, sizeof(int) * M_SIZE * M_SIZE);
    cudaMalloc((void **)&gMatResult, sizeof(int) * M_SIZE * M_SIZE);
    //GlobalMemory
    cudaMemcpy(gMat1, mat1, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gMat2, mat2, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gMatResult, matResult, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
    //CPU
    cudaEvent_t cpuStartTime,cpuStopTime;
    float cpuTime;
    cudaEventCreate(&cpuStartTime);
    cudaEventCreate(&cpuStopTime);
    cudaEventRecord(cpuStartTime, 0);
    matMul(mat1, mat2, matResult);
    cudaEventRecord(cpuStopTime, 0);
    cudaEventSynchronize(cpuStopTime);
    cudaEventElapsedTime(&cpuTime, cpuStartTime, cpuStopTime);
    //CUDA
    cudaEvent_t cudaStartTime, cudaStopTime;
    float cudaTime;
    dim3 Dg(M_SIZE*M_SIZE/B_SIZE, 1, 0), Db(B_SIZE, 1, 1);
    cudaEventCreate(&cudaStartTime);
    cudaEventCreate(&cudaStopTime);
    cudaEventRecord(cudaStartTime, 0);
    cudaMatMulSharedTsukuba <<< Dg, Db>>> (gMat1, gMat2, gMatResult);
    cudaEventRecord(cudaStopTime, 0);
    cudaEventSynchronize(cudaStopTime);
    cudaEventElapsedTime(&cudaTime, cudaStartTime, cudaStopTime);
    //Cuda
    int *cudamat1 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *cudamat2 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *cudamat3 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    cudaMemcpy(cudamat1, gMat1, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(cudamat2, gMat2, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(cudamat3, gMatResult, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
    //¿¿
    printf("M_SIZE:%d\n",M_SIZE);
    if(matDiffCount(mat1,cudamat1)){
        puts("CPU¤ÈGPU¤Î·×»»·ë²Ì¤Ï°ìÃ×¤·¤Þ¤·¤¿");
    }else{
        puts("CPU¤ÈGPU¤Î·×»»·ë²Ì¤Ï°ìÃ×¤·¤Þ¤»¤ó¤Ç¤·¤¿");
    }
    if(M_SIZE < 10){
        puts("M1");
        printMat(mat1);
        puts("M2");
        printMat(mat2);
        puts("M3");
        printMat(matResult);
        puts("cudaM");
        printMat(cudamat1);
        puts("cuda2");
        printMat(cudamat2);
        puts("cuda3");
        printMat(cudamat3);
    }
    printf("CPU:Time = %f\n", cpuTime);
    printf("GPU:Time = %f\n", cudaTime);
    //Host Memory
    free(mat1);
    free(mat2);
    free(matResult);
    //Global Memory
    cudaFree(gMat1);
    cudaFree(gMat2);
    cudaFree(gMatResult);
    return 0;
}
//CPU
void matMul(int *mat1, int *mat2, int *dstMat){
    int transMat[M_SIZE*M_SIZE];
    matZeros(transMat);
    matTrans(mat2,transMat);//
    for(int i=0; i<M_SIZE; i++){
        for(int j=0; j<M_SIZE; j++){
            for(int k=0; k<M_SIZE; k++){
                dstMat[i*M_SIZE + j] += mat1[i*M_SIZE + k] * transMat[j*M_SIZE + k];
            }
        }
    }
}
//CUDA
__global__ void cudaMatMul(int *mat1, int *mat2, int *dstMat){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int row = id/M_SIZE;
    int column = id%M_SIZE;
    int x=0;
    for(int i=0; i<M_SIZE; i++){
        x += mat1[row*M_SIZE+i] * mat2[i*M_SIZE+column];
    }
    dstMat[id] = x;
}
//CUDA,SharedMemory
__global__ void cudaMatMulShared(int *mat1, int *mat2, int *dstMat){
    __shared__ int sMat1[1024], sMat2[1024];
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    sMat1[tid] = mat1[id];
    sMat2[tid] = mat2[id];
    __syncthreads();
    int row = id/M_SIZE;
    int column = id%M_SIZE;
    int x=0;
    for(int i=0; i<M_SIZE; i++){
        x += sMat1[row*M_SIZE+i] * sMat2[i*M_SIZE+column];
    }
    dstMat[id] = x;
}
//tsukuba,Shared
__global__ void cudaMatMulSharedTsukuba(int *mat1, int *mat2, int *dstMat){
    __shared__ int sMat1[1024],sMat2[1024];
    unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    //SM1[tid]     = gMat1[id];
    //SM1[tid+512] = gMat1[id+512];
    sMat1[tid]     = mat1[id];
    sMat1[tid+512] = mat1[id+512];
    sMat2[tid]     = mat2[id];
    sMat2[tid+512] = mat2[id+512];
    __syncthreads();
    int row = id/M_SIZE;
    int column = id %M_SIZE;
    int x = 0;
    for(int i=0; i<M_SIZE; i++){
        x += sMat1[row * M_SIZE + i] * sMat2[i * M_SIZE + column];
    }
    mat1[id] = x;
}
//
void matZeros(int *mat){
    for(int i=0; i<M_SIZE*M_SIZE; i++){
        mat[i]=0;
    }
}
//
void matTrans(int *mat, int *dstMat){
    for(int i=0; i<M_SIZE; i++){
        for(int j=0; j<M_SIZE; j++){
            dstMat[j*M_SIZE + i] = mat[i*M_SIZE + j];
        }
    }
}
//
int matDiffCount(int *mat1, int *mat2){
    int count = 0;
    for(int i=0; i<M_SIZE*M_SIZE; i++){
        if(mat1[i] - mat2[i]){
            return 0;
        }else{
            count++;
        }
    }
    return count;
}
void printMat(int *mat){
    for(int i=0; i<M_SIZE; i++){
        for(int j=0; j<M_SIZE;j++){
            printf("%d,",mat[i*M_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}
