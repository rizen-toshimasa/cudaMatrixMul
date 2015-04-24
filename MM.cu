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
void printHMNum(int *HM, int num);
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
        HM2[i] = 1;//rand()%2;
        HM3[i] = 1;//rand()%2;
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
    cudaMatrixMulShared <<< Dg, Db>>> (GM1, GM2, GM3);
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
    if(1){
        puts("M1");
        printHMNum(HM1,10);
        puts("M2");
        printHMNum(HM2,10);
        puts("M3");
        printHMNum(HM3,10);
        puts("cudaM");
        printHMNum(cudaHM1, 10);
        puts("cuda2");
        printHMNum(cudaHM2, 10);
        puts("cuda3");
        printHMNum(cudaHM3, 10);
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
    for(int i=0; i < SUB_SIZE; i++){
        SM2[tid + M_SIZE * i] = GM2[tid + M_SIZE * i + blockIdx.y * M_SIZE * SUB_SIZE];
        SM3[tid + M_SIZE * i] = GM3[tid + M_SIZE * i + blockIdx.x * M_SIZE * SUB_SIZE];
    }
    __syncthreads();
    if(threadIdx.x <=10 && blockIdx.x == 0 && blockIdx.y == 0){
        printf("tid=%d, GM2[0]=%d SM2[0]=%d\n",tid,GM2[0],SM2[0]);
        printf("tid=%d, GM2[1]=%d SM2[1]=%d\n",tid,GM2[1],SM2[1]);
        printf("tid=%d, GM2[2]=%d SM2[2]=%d\n",tid,GM2[2],SM2[2]);
        printf("tid=%d, GM3[0]=%d SM3[0]=%d\n",tid,GM3[0],SM3[0]);
        printf("tid=%d, GM3[1]=%d SM3[1]=%d\n",tid,GM3[1],SM3[1]);
        printf("tid=%d, GM3[2]=%d SM3[2]=%d\n",tid,GM3[2],SM3[2]);
       
    }
    __syncthreads();
    for(int i=0; i < SUB_SIZE; i++){
        for(int j=0; j < SUB_SIZE; j++){
            //総和をとる
            //今はGMに直接足しこんでいるが,内部的にはレジスタに取り込んで
            //足して戻してを繰り返していると思われ,非効率である
            //SMにおいて総和するか,
            //GMにおいてSMに戻して総和にするかは後で考える
            
            //__syncthreads();
            GM1[M_SIZE*(blockIdx.y*SUB_SIZE + i) + blockIdx.x*SUB_SIZE + j] += SM2[i*SUB_SIZE + tid] * SM3[j*SUB_SIZE + tid];
            //__syncthreads();
        }
    }
    __syncthreads();
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
void printHMNum(int *HM, int num){
    for(int i=0; i<num; i++){
        for(int j=0; j<num;j++){
            printf("%d,",HM[i*M_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}
