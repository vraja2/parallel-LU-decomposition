#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>

double** generate_matrix(int dim) {
  double **matrix = (double**) malloc(dim*sizeof(double*));
  int i,j;
  for(i=0;i<dim;i++) {
    matrix[i] = (double*) malloc(dim*sizeof(double));
  }
  // initialize matrix to random values
  for(i=0;i<dim;i++) {
    for(j=0;j<dim;j++) {
      matrix[i][j] = (float) (rand() % 20)+1;
    }
  }
  return matrix; 
}

void free_matrix(int dim, double** matrix) {
  int i;
  for(i=0;i<dim;i++) {
    free(matrix[i]);
  }
  free(matrix);
}

void print_matrix(int dim, double** matrix) {
  // somewhat of a hacky way of printing a matrix with alignment, need to fix for negatives
  int i,j;
  for(i=0;i<dim;i++) {
    for (j=0;j<dim;j++) {
      if (abs(matrix[i][j]) < 10) {
        printf("%f     ",matrix[i][j]);
      } else if (abs(matrix[i][j]) < 100) {
        printf("%f    ", matrix[i][j]);
      } else {
        printf("%f  ", matrix[i][j]);
      }
    }
    printf("\n");
  }
  printf("\n");
}

double **create_zero_matrix(int dim) {
  double **matrix = (double**) malloc(dim*sizeof(double*));
  int i,j;
  for(i=0;i<dim;i++) {
    matrix[i] = (double*) malloc(dim*sizeof(double));
  }
  for(i=0;i<dim;i++) {
    for(j=0;j<dim;j++) {
      matrix[i][j] = 0.;
    }
  }
  return matrix;
}

void parallel_lu(int argc, char **argv, double **matrix, int dim, int block_dim) {
  int procs;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  double **L = create_zero_matrix(dim);

  int proc_per_row = dim/block_dim; 
  int col_start = (rank*block_dim) % dim;
  int col_end = col_start+block_dim-1;
  int row_start = (rank/proc_per_row)*block_dim;
  int row_end = row_start+block_dim-1;

  printf("Rank %i\n", rank);
  printf("Col start %i\n", col_start);
  printf("Col end %i\n", col_end);
  printf("Row start %i\n", row_start);
  printf("Row end %i\n", row_end);
  
  int k,i;
  for(k=0;k<dim;k++) {
    if(k>=col_start && k<=col_end) {
      for(i=row_start;i<=row_end;i++) {
        if (i>k) {
          L[i][k] = matrix[i][k]/matrix[k][k];
        }
      }
    }
  }
  // TODO: send/recv ratios and figure out the proper data size to send/recv. 
  // if k is a column for current process, send ratios
  if (k>=col_start && k<=col_end) {
  
  }
  int j;
  for (j=col_start;j<=col_end;j++) {
    if (j>k) {
      for (i=row_start;i<=row_end;i++) {
        matrix[i][j] = matrix[i][j]-L[i][k]*matrix[k][j];
      }
    }
  }
  print_matrix(dim,L);
  MPI_Finalize();
}

int main(int argc, char **argv) {
  int dims[3] = {5,10,15};
  int dim = 4;
  int block_dim = 2;
  parallel_lu(argc, argv, generate_matrix(dim), dim, block_dim);
  return 0;
}
