#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

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

double **create_perm_matrix(double **matrix, int dim) {
  double **P = create_zero_matrix(dim);
  // if a row was already found to have a max element, it's flag in this array will be 1.
  int *seen_rows = (int *) malloc(dim*sizeof(int));
  int i,j;
  for(i=0;i<dim;i++) {
    seen_rows[i] = 0;
  }
  for(i=0;i<dim;i++) {
    int min_index = -1;
    int min_val = INT_MAX;
    for(j=0;j<dim;j++) {
      if (!seen_rows[j] && min_val > abs(matrix[j][i])) {
        min_val = matrix[j][i];
        min_index = j;
      }
    }
    seen_rows[min_index] = 1;
    P[i][min_index] = 1.;
  }
  return P;
}

// currently only supporting equal sized square matrices for purposes of LU decomposition
double **matrix_mult(double **A, double **B, int dim) {
  double **result = create_zero_matrix(dim);
  int i,j,k;
  for(i=0;i<dim;i++) {
    for(j=0;j<dim;j++) {
      for(k=0;k<dim;k++) {
        result[i][j] += A[i][k]*B[k][j]; 
      } 
    }
  }
  return result;
}

void serial_lu(double **matrix, int dim) {
  double **P = create_perm_matrix(matrix,dim);
  double **permuted_mat = matrix_mult(P,matrix,dim);
  double **L = create_zero_matrix(dim);
  /* 
  printf("A=\n");
  print_matrix(dim,permuted_mat);
  */
  int i,j,k;
  // initialize diagonal of L to 1
  for(i=0;i<dim;i++) {
    L[i][i] = 1;
  }

  // columns
  for(j=0;j<dim;j++) {
    // rows
    for(i=j+1;i<dim;i++) {
      float ratio = permuted_mat[i][j]/permuted_mat[j][j];
      L[i][j] = ratio;
      for(k=0;k<dim;k++) {
        permuted_mat[i][k] -= ratio*permuted_mat[j][k];
      }
    }
  }
  
  /*
  printf("L=\n");
  print_matrix(dim,L);
  printf("U=\n");
  print_matrix(dim,permuted_mat);
  printf("L*U=\n");
  print_matrix(dim,matrix_mult(L,permuted_mat,dim));
  */
}

void run_decompositions(int *dims, int n) {
  int i;
  double **matrix;
  for(i=0;i<n;i++) {
    matrix = generate_matrix(dims[i]);
    clock_t begin, end;
    begin = clock();
    serial_lu(matrix,dims[i]);
    end = clock();
    double time_spent = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Took %f seconds for dim=%i\n",time_spent,dims[i]);
    free_matrix(dims[i],matrix);
  } 
}

int main() {
  int dim = 10; 
  double **matrix = generate_matrix(dim);
  int dims[3] = {5,10,15};
  run_decompositions(dims,3);
  free_matrix(dim,matrix);
  return 0;
}
