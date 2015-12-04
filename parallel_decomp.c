#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define rank2print 0
#define doSerial 0

double** generate_matrix(int dim) {
  double **matrix = (double**) malloc(dim*sizeof(double*));
  int i,j;
  for(i=0;i<dim;i++) {
    matrix[i] = (double*) malloc(dim*sizeof(double));
  }
  // initialize matrix to random values
  srand(2);
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

void print_vector(int dim, double* vector) {
  int i;
  for(i=0;i<dim;i++) {
      if (abs(vector[i]) < 10) {
        printf("%f     ",vector[i]);
      } else if (abs(vector[i]) < 100) {
        printf("%f    ", vector[i]);
      } else {
        printf("%f  ", vector[i]);
      }
  }
  printf("\n");
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
  MPI_Status status;
  MPI_Request request;

  //Check if number of processes is a perfect square
  if (sqrt(procs)!=floor(sqrt(procs))) {
	  printf("Number of processes is not a perfect square!\n");
	  return;
  }
  int num_rows = sqrt(procs);
  int num_cols = sqrt(procs);

  int dimSize[2] = {num_rows, num_cols};
  int periodic[2] = {0, 0};
  int myCoords[2];

  MPI_Comm comm2D;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, periodic, 0, &comm2D);

  int myRow, myCol;
  MPI_Cart_coords(comm2D, rank, 2, myCoords);
  myRow = myCoords[0]; myCol = myCoords[1];

  //Determine the neighbour rank numbers
  int rightRank;
  int leftRank = rank;
  int botRank;
  int topRank = rank;
  MPI_Cart_shift(comm2D, 1, 1, &leftRank, &rightRank);
  MPI_Cart_shift(comm2D, 0, 1, &topRank, &botRank);
  
  double **L = create_zero_matrix(dim);
  double *LBuffSend = (double*) malloc (block_dim * sizeof(double));
  double *LBuffRecv = (double*) malloc (block_dim * sizeof(double));
  double *PBuffSend = (double*) malloc (block_dim * sizeof(double));
  double *PBuffRecv = (double*) malloc (block_dim * sizeof(double));

  int proc_per_row = dim/block_dim; 
  int col_start = (rank*block_dim) % dim;
  int col_end = col_start+block_dim-1;
  int row_start = (rank/proc_per_row)*block_dim;
  int row_end = row_start+block_dim-1;

  if(rank==rank2print) {
	printf("Rank %i\n", rank);
	printf("myRow of proc:%i\n", myRow);
	printf("myCol of proc:%i\n", myCol);
	printf("Right rank is: %i\n",rightRank);
	printf("Left rank is: %i\n",leftRank);
	printf("Top rank is: %i\n",topRank);
	printf("Bottom rank is: %i\n",botRank);
	printf("Col start %i\n", col_start);
	printf("Col end %i\n", col_end);
	printf("Row start %i\n", row_start);
	printf("Row end %i\n", row_end);
	print_matrix(dim,matrix);
  }
  
  int i,j,k;
  for(k=0;k<dim;k++) {
	//Send & recieve pivot row
	//Recieve PBuffRec from top and send PBuffSend to bottom (order here is important)
	bool kInMyRows = (k>=row_start && k<=row_end);
	bool kInTopRows = (k>=(row_start-block_dim) && k<=(row_end-block_dim));
	if(topRank != -1 && kInTopRows) {
		MPI_Recv(PBuffRecv, block_dim, MPI_DOUBLE, topRank, 0, MPI_COMM_WORLD, &status);
		if(rank==rank2print) {
			printf("Recieved pivot row from rank %i: ",topRank);
			print_vector(block_dim,PBuffRecv);
		}
		//Place PBuffRecv in correct place of matrix
		for(j=col_start;j<=col_end;j++) {
			if(j>=k) {
				matrix[k][j] = PBuffRecv[j-col_start];
			}
		}
	}
	if(botRank != -1 && kInMyRows) {
		//Assemble PBuffSend
		for(j=col_start;j<=col_end;j++) {
			if(j>=k) {
				PBuffSend[j-col_start] = matrix[k][j];
			}
		}
		if(rank==rank2print) {
			printf("Sending pivot row to rank %i: ",botRank);
			print_vector(block_dim,PBuffSend);
		}
		MPI_Isend(PBuffSend, block_dim, MPI_DOUBLE, botRank, 0, MPI_COMM_WORLD, &request);
	} 

	if(rank==rank2print) {
		printf("U:\n");
		print_matrix(dim,matrix);
	}

	//Calculate ratios
	bool kInMyCols = (k>=col_start && k<=col_end);
    if(kInMyCols) {
      for(i=row_start;i<=row_end;i++) {
        if (i>k) {
          L[i][k] = matrix[i][k]/matrix[k][k];
        }
      }
    }

	//Wait for PBuffSend to be usable
	if(botRank != -1 && kInMyRows)
		MPI_Wait(&request, &status);

	if(rank==rank2print) {
		printf("L:\n");
		print_matrix(dim,L);
	}
	
	//Send & recieve ratios
	//Recieve LBuffRec from left and send LBuffSend to right (order here is important)
	bool kInLeftCols = (k>=(col_start-block_dim) && k<=(col_end-block_dim));
	if(leftRank != -1 && kInLeftCols) {
		MPI_Recv(LBuffRecv, block_dim, MPI_DOUBLE, leftRank, 0, MPI_COMM_WORLD, &status);
		if(rank==rank2print) {
			printf("Recieved L from rank %i: ",leftRank);
			print_vector(block_dim,LBuffRecv);
		}
		//Place LBuffRecv in correct place of L[i][k]
		for(i=row_start;i<=row_end;i++) {
			if(i>k) {
				L[i][k] = LBuffRecv[i-row_start];
			}
		}
	}
	if(rightRank != -1 && kInMyCols) {
		//Assemble LBuffSend
		for(i=row_start;i<=row_end;i++) {
			if(i>k) {
				LBuffSend[i-row_start] = L[i][k];
			}
		}
		if(rank==rank2print) {
			printf("Sending L to rank %i: ",rightRank);
			print_vector(block_dim,LBuffSend);
		}
		MPI_Isend(LBuffSend, block_dim, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, &request);
	}
  

  // TODO: send/recv ratios and figure out the proper data size to send/recv. 
  // if k is a column for current process, send ratios
  //if (k>=col_start && k<=col_end) {
  //}

	//Compute upper triangular matrix
	for (j=col_start;j<=col_end;j++) {
		if (j>k) {
			for (i=row_start;i<=row_end;i++) {
				if (i>k) {
					matrix[i][j] = matrix[i][j]-L[i][k]*matrix[k][j];
				}
			}
		}
	}
	//Wait for LBuffSend to be usable
	if(rightRank != -1 && kInMyCols)
		MPI_Wait(&request, &status);
  }
  if(rank==rank2print)
	print_matrix(dim,matrix);
  MPI_Finalize();
  free_matrix(dim,L);
  free_matrix(dim,matrix);
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
  double **L = create_zero_matrix(dim);
  
  printf("A=\n");
  print_matrix(dim,matrix);

  int i,j,k;
  // initialize diagonal of L to 1
  for(i=0;i<dim;i++) {
    L[i][i] = 1;
  }

  // columns
  for(j=0;j<dim;j++) {
    // rows
    for(i=j+1;i<dim;i++) {
      float ratio = matrix[i][j]/matrix[j][j];
      L[i][j] = ratio;
      for(k=0;k<dim;k++) {
        matrix[i][k] -= ratio*matrix[j][k];
      }
    }
  }
  
  
  printf("L=\n");
  print_matrix(dim,L);
  printf("U=\n");
  print_matrix(dim,matrix);
  printf("L*U=\n");
  print_matrix(dim,matrix_mult(L,matrix,dim));
  free_matrix(dim,L);
  free_matrix(dim,matrix);
}


int main(int argc, char **argv) {
  int dims[3] = {5,10,15};
  int dim = 4;
  int block_dim = 2;
  if(doSerial==1) {
	serial_lu(generate_matrix(dim),dim);
	printf("Done with serial.....");
  }
  parallel_lu(argc, argv, generate_matrix(dim), dim, block_dim);
  return 0;
}
