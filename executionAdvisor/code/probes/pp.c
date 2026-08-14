#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
int main(int c,char**v){
  int r; MPI_Init(&c,&v); MPI_Comm_rank(MPI_COMM_WORLD,&r);
  char *b = malloc(64L<<20);
  if(!b){ if(!r) fprintf(stderr,"alloc failed\n"); MPI_Abort(MPI_COMM_WORLD,1); }
  if(!r) printf("%12s %12s %12s\n","bytes","lat_us","MB/s");
  for(long n=8; n<=(64L<<20); n*=8){
    int it = n<65536 ? 2000 : 50;
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    for(int i=0;i<it;i++){
      if(r==0){ MPI_Send(b,n,MPI_BYTE,1,0,MPI_COMM_WORLD);
                MPI_Recv(b,n,MPI_BYTE,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
      else    { MPI_Recv(b,n,MPI_BYTE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Send(b,n,MPI_BYTE,0,0,MPI_COMM_WORLD); }
    }
    t = MPI_Wtime()-t;
    if(!r) printf("%12ld %12.1f %12.2f\n", n, t/it/2*1e6, (double)n*it*2/t/1e6);
  }
  free(b); MPI_Finalize(); return 0;
}
