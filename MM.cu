#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <cblas.h>//cuda用線形計算ライブラリ 
//static const int M_SIZE = 3;//matrix size
//static const int B_SIZE = 1024;//block size
#define M_SIZE 1024
#define B_SIZE 1024
#define SUB_SIZE 6
//CPU プロトタイプ
void matrixMul(int *HM1, int *HM2, int *HM3);
void matrixZeros(int *HM);
void matrixTranspose(int *iMat, int*oMat);
int matrixDiffCount(int *HM1, int *HM2);
void printHM(int *HM);
//CUDA プロトタイプ
__global__ void cudaMatrixMul(int *GM1, int *GM2, int *GM3);
__global__ void cudaMatrixMulShared(int *GM1, int *GM2, int *GM3);
//メイン関数
int main(void){
    srand((unsigned)time(NULL));
    //Host Memory に Matrix 確保 HM1 = HM2 * HM3
    int *HM1 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *HM2 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *HM3 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    //Host Memoryにデータ格納
    matrixZeros(HM1);
    for(int i=0; i<M_SIZE*M_SIZE; i++){
        HM2[i] = rand()%256;
        HM3[i] = rand()%256;
    }
    //Global Memory 確保
    int *GM1,*GM2,*GM3;
    cudaMalloc((void **)&GM1, sizeof(int) * M_SIZE * M_SIZE);
    cudaMalloc((void **)&GM2, sizeof(int) * M_SIZE * M_SIZE);
    cudaMalloc((void **)&GM3, sizeof(int) * M_SIZE * M_SIZE);
    //GlobalMemoryにデータ格納
    cudaMemcpy(GM1, HM1, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(GM2, HM2, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
    int *TM = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    matrixTranspose(HM3,TM);
    cudaMemcpy(GM3, TM, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
    //CPUでの計算
    cudaEvent_t cpuStartTime,cpuStopTime;
    float cpuTime;
    cudaEventCreate(&cpuStartTime);
    cudaEventCreate(&cpuStopTime);
    cudaEventRecord(cpuStartTime, 0);
    matrixMul(HM1, HM2, HM3);
    cudaEventRecord(cpuStopTime, 0);
    cudaEventSynchronize(cpuStopTime);
    cudaEventElapsedTime(&cpuTime, cpuStartTime, cpuStopTime);
    //CUDAでの計算
    cudaEvent_t cudaStartTime, cudaStopTime;
    float cudaTime;
    dim3 Dg(M_SIZE/SUB_SIZE, M_SIZE/SUB_SIZE, 1), Db(1024, 1, 1);
    cudaEventCreate(&cudaStartTime);
    cudaEventCreate(&cudaStopTime);
    cudaEventRecord(cudaStartTime, 0);
    cudaMatrixMulShared <<< Dg, Db, 40000>>> (GM1, GM2, GM3);
    cudaEventRecord(cudaStopTime, 0);
    cudaEventSynchronize(cudaStopTime);
    cudaEventElapsedTime(&cudaTime, cudaStartTime, cudaStopTime);
    //Cuda計算結果をHostMemoryにコピー
    int *cudaHM1 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *cudaHM2 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    int *cudaHM3 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
    cudaMemcpy(cudaHM1, GM1, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(cudaHM2, GM2, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(cudaHM3, GM3, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
    //標準出力
    printf("M_SIZE:%d\n",M_SIZE);
    if(matrixDiffCount(HM1,cudaHM1)){
        puts("CPUとGPUの計算結果は一致しました");
    }else{
        puts("CPUとGPUの計算結果は一致しませんでした");
    }
    if(M_SIZE <= 10){
        puts("M1");
        printHM(HM1);
        puts("M2");
        printHM(HM2);
        puts("M3");
        printHM(HM3);
        puts("cudaM");
        printHM(cudaHM1);
        puts("cuda2");
        printHM(cudaHM2);
        puts("cuda3");
        printHM(cudaHM3);
    }
    printf("CPU:Time = %f\n", cpuTime);
    printf("GPU:Time = %f\n", cudaTime);
    //Host Memory開放
    free(HM1);
    free(HM2);
    free(HM3);
    //Global Memory開放
    cudaFree(GM1);
    cudaFree(GM2);
    cudaFree(GM3);
    return 0;
}
//CUDA版行列の積
__global__ void cudaMatrixMul(int *GM1, int *GM2, int *GM3){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int row = id/M_SIZE;
    int column = id%M_SIZE;
    int x=0;
    for(int i=0; i<M_SIZE; i++){
        x += GM2[row*M_SIZE+i] * GM3[i*M_SIZE+column];
    }
    GM1[id] = x;
}
//CUDA,SharedMemory使用版行列の積
__global__ void cudaMatrixMulShared(int *GM1, int *GM2, int *GM3){
    __shared__ int SM2[M_SIZE*SUB_SIZE], SM3[M_SIZE*SUB_SIZE];
    unsigned int tid = threadIdx.x;
    //ここらへんに転置する処理
    //GlobalMem -> SharedMem
    for(int i=SUB_SIZE; i < SUB_SIZE; i++){
        SM2[tid + M_SIZE * i] = GM2[tid + M_SIZE * i + blockIdx.y * M_SIZE * SUB_SIZE];
        SM3[tid + M_SIZE * i] = GM3[tid + M_SIZE * i + blockIdx.x * M_SIZE * SUB_SIZE];
    }
    __syncthreads();
    if(blockIdx.x == 0 && blockIdx.y == 0){
        printf("SM2[0]=%d, ",SM2[0]);
        printf("SM2[1]=%d, ",SM2[1]);
        printf("SM2[2]=%d\n",SM2[2]);
        printf("SM3[0]=%d, ",SM3[0]);
        printf("SM3[1]=%d, ",SM3[1]);
        printf("SM3[2]=%d\n",SM3[2]);
       
    }
    __syncthreads();
    for(int i=SUB_SIZE; i < SUB_SIZE; i++){
        for(int j=SUB_SIZE; j < SUB_SIZE; j++){
            //総和をとる
            //今はGMに直接足しこんでいるが,内部的にはレジスタに取り込んで
            //足して戻してを繰り返していると思われ,非効率である
            //SMにおいて総和するか,
            //GMにおいてSMに戻して総和にするかは後で考える
            GM1[M_SIZE*(blockIdx.y*SUB_SIZE + i) + blockIdx.x*SUB_SIZE + j] += SM2[i*SUB_SIZE + tid] * SM3[j*SUB_SIZE + tid];
        }
    }
}
//CPU版行列の積
void matrixMul(int *HM1, int *HM2, int *HM3){
    int tmpHM[M_SIZE*M_SIZE];
    matrixZeros(tmpHM);
    matrixTranspose(HM3,tmpHM);//転置
    for(int i=0; i<M_SIZE; i++){
        for(int j=0; j<M_SIZE; j++){
            for(int k=0; k<M_SIZE; k++){
                HM1[i*M_SIZE + j] += HM2[i*M_SIZE + k] * tmpHM[j*M_SIZE + k];
            }
        }
    }
}
void matrixZeros(int *HM){
    for(int i=0; i<M_SIZE*M_SIZE; i++){
        HM[i]=0;
    }
}
//転置行列
void matrixTranspose(int *iMat, int *oMat){
    for(int i=0; i<M_SIZE; i++){
        for(int j=0; j<M_SIZE; j++){
            oMat[j*M_SIZE + i] = iMat[i*M_SIZE + j];
        }
    }
}
int matrixDiffCount(int *HM1, int *HM2){
    for(int i=0; i<M_SIZE*M_SIZE; i++){
        if(HM1[i] - HM2[i]){
            return 0;
        }
    }
    return 1;
}
void printHM(int *HM){
    for(int i=0; i<M_SIZE; i++){
        for(int j=0; j<M_SIZE;j++){
            printf("%d,",HM[i*M_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}
