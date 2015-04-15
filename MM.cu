//正方行列の積を求め.cpuとgpuで速度比較する
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M_SIZE 1024//matrix size
#define B_SIZE 256//block size

//CPU プロトタイプ
void matrixMul(int *HM1, int *HM2, int *HM3);
void matrixZeros(int *HM);
int matrixDiffCount(int *HM1, int *HM2);
__global__ void cudaMatrixMulShared(int *GM1, int *GM2, int *GM3);
void printHM(int *HM);
//CUDA プロトタイプ
__global__ void cudaMatrixMul(int *GM1, int *GM2, int *GM3);

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
  cudaMemcpy(GM3, HM3, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);

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
  dim3 Dg(M_SIZE * M_SIZE/B_SIZE, 1, 1), Db(B_SIZE, 1, 1);
  cudaEventCreate(&cudaStartTime);
  cudaEventCreate(&cudaStopTime);
  cudaEventRecord(cudaStartTime, 0);
  cudaMatrixMul <<< Dg, Db>>> (GM1, GM2, GM3);
  cudaEventRecord(cudaStopTime, 0);
  cudaEventSynchronize(cudaStopTime);
  cudaEventElapsedTime(&cudaTime, cudaStartTime, cudaStopTime);
  
  //Cuda計算結果をHostMemoryにコピー
  int *cudaHM = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
  cudaMemcpy(cudaHM, GM1, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyDeviceToHost);
  
  //標準出力
  printf("M_SIZE:%d\n",M_SIZE);
  if(matrixDiffCount(HM1,cudaHM)){
    puts("CPUとGPUの計算結果は一致しました");
  }else{
    puts("CPUとGPUの計算結果は一致しませんでした");
  }
  if(M_SIZE < 10){
    puts("M1");
    printHM(HM1);
    puts("M2");
    printHM(HM2);
    puts("M3");
    printHM(HM3);
    printf("cudaM");
    printHM(cudaHM);
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
  
}

//CPU版行列の積
void matrixMul(int *HM1, int *HM2, int *HM3){
  for(int i=0; i<M_SIZE; i++){
    for(int j=0; j<M_SIZE; j++){
      for(int k=0; k<M_SIZE; k++){
	HM1[i*M_SIZE + j] += HM2[i*M_SIZE + k] * HM3[k*M_SIZE + j];
      }
    }
  }
}


void matrixZeros(int *HM){
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    HM[i]=0;
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