#include <stdio.h>
#include <stdlib.h>
#define M_SIZE 3
int main(void){
  srand(123);
  int *MM1 = (int *)malloc(M_SIZE * M_SIZE);
  int *MM2 = (int *)malloc(M_SIZE * M_SIZE);
  int *MM3 = (int *)malloc(M_SIZE * M_SIZE);
  
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    MM1[i] = rand()%10;
    MM2[i] = rand()%10;

  }
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    printf("M1:%d ",MM1[i]);
  }
  puts("");
  for(int i=0; i<M_SIZE*M_SIZE; i++){
    printf("M2:%d ",MM2[i]);
  }
  
  return 0;
}
