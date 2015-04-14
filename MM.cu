//正方行列の積を求め.cpuとgpuで速度比較する
#include <stdio.h>
#include <stdlib.h>
#define M_SIZE 3//matrix size
void matrixMul(int *HM1, int *HM2, int *HM3){
  for(int i=0; i<M_SIZE; i++){
    for(int j=0; j<M_SIZE; j++){
      for(int k=0; k<M_SIZE; k++){
	HM1[i*M_SIZE + j] += HM2[i*M_SIZE + k] * HM3[k*M_SIZE + j];
      }
    }
  }
}
void zerosM(int *HM){
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    HM[i]=0;
  }
}
void printHM(int *HM){
  for(int i=0; i<M_SIZE; i++){
    for(int j=0; j<M_SIZE;j++){
      printf("%d,",HM[i*M_SIZE + j]);
    }
    printf("\n");
  }
  printf("\n");
}/*
__global__ void matrixMulGPU(int *GM1, int GM2, int GM3){
  int id = blockDimxx * blockIdx.x + threadIdx.x;
 
  }*/
int main(void){
  srand(123);
  //Host Memory に Matrix 確保 HM1 = HM2 * HM3
  int *HM1 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
  int *HM2 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
  int *HM3 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);

  //Host Memoryにデータ格納
  zerosM(HM1);
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    HM2[i] = rand()%10;
    HM3[i] = rand()%10;
  }
  //Global Memory 確保
  int *GM1,*GM2,*GM3;
  cudaMalloc((void **)&GM1, sizeof(int) * M_SIZE * M_SIZE);
  cudaMalloc((void **)&GM2, sizeof(int) * M_SIZE * M_SIZE);
  cudaMalloc((void **)&GM3, sizeof(int) * M_SIZE * M_SIZE);

  //
  matrixMul(HM1, HM2, HM3);
  puts("matrix multiplication(M1 = M2 * M3)");
  puts("M1");
  printHM(HM1);
  puts("M2");
  printHM(HM2);
  puts("M3");
  printHM(HM3);

  free(HM1);
  free(HM2);
  free(HM3);
  return 0;
}
