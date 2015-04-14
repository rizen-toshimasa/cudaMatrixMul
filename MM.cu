//����������Ѥ���.cpu��gpu��®����Ӥ���
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M_SIZE 1024//matrix size

//�ץ�ȥ�����
void matrixMul(int *HM1, int *HM2, int *HM3);
void zerosM(int *HM);
void printHM(int *HM);
void fprintM(char *fileName,int *HM);
__global__ void cudaMatrixMul(int *GM1, int *GM2, int *GM3);

//�ᥤ��ؿ�
int main(void){
  srand(123);
  //Host Memory �� Matrix ���� HM1 = HM2 * HM3
  int *HM1 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
  int *HM2 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);
  int *HM3 = (int *)malloc(sizeof(int) * M_SIZE * M_SIZE);

  //Host Memory�˥ǡ�����Ǽ
  zerosM(HM1);
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    HM2[i] = rand()%10;
    HM3[i] = rand()%10;
  }

  //Global Memory ����
  int *GM1,*GM2,*GM3;
  cudaMalloc((void **)&GM1, sizeof(int) * M_SIZE * M_SIZE);
  cudaMalloc((void **)&GM2, sizeof(int) * M_SIZE * M_SIZE);
  cudaMalloc((void **)&GM3, sizeof(int) * M_SIZE * M_SIZE);
  
  //GlobalMemory�˥ǡ�����Ǽ
  cudaMemcpy(GM1, HM1, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(GM2, HM2, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(GM3, HM3, sizeof(int) * M_SIZE * M_SIZE, cudaMemcpyHostToDevice);

  //CPU�Ǥη׻�
  clock_t startTime,stopTime;
  startTime = clock();
  matrixMul(HM1, HM2, HM3);
  stopTime = clock();
  
  //CUDA�Ǥη׻�
  cudaEvent_t cudaStartTime, cudaStopTime;
  float cudaTime;
  dim3 Dg(5, 5, 1), Db(4, 4, 2);
  cudaEventCreate(&cudaStartTime);
  cudaEventCreate(&cudaStopTime);
  cudaEventRecord(cudaStartTime, 0);
  cudaMatrixMul <<< Dg, Db>>> (GM1, GM2, GM3);
  cudaEventRecord(cudaStopTime, 0);
  cudaEventSynchronize(cudaStopTime);
  cudaEventElapsedTime(&cudaTime, cudaStartTime, cudaStopTime);

  //ɸ�����
  /*
  puts("M1");
  printHM(HM1);
  puts("M2");
  printHM(HM2);
  puts("M3");
  printHM(HM3);*/
  printf("CPU:Time = %f\n", (double)(stopTime - startTime)/CLOCKS_PER_SEC);
  printf("GPU:Time = %f\n", cudaTime);
  
 
  fprintM("cpuMatrixMul.txt",HM1);
  
  //Host Memory����
  free(HM1);
  free(HM2);
  free(HM3);
  
  //Global Memory����
  cudaFree(GM1);
  cudaFree(GM2);
  cudaFree(GM3);
  return 0;
}

//CUDA�ǹ������
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

//CPU�ǹ������
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
}
void fprintM(char *fileName,int *HM){
  FILE *fp;
  fp = fopen(fileName, "w");
  for(int i=0; i<M_SIZE; i++){
    for(int j=0; j<M_SIZE; j++){
      fscanf(fp, "%d ", &HM);
    }
    puts("");
  }
  puts("");
}