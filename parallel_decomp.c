#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define doPAPI 0

#if doPAPI==1
#include "support.h"
#endif

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
			matrix[i][j] = (double) (rand() % 20)+1;
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

void print_matrix_chunk(int dim, int row_start, int col_start, double** matrix) {
	// somewhat of a hacky way of printing a matrix with alignment, need to fix for negatives
	// Only for square matrix chunks
	int i,j;
	for(i=row_start;i<(row_start+dim);i++) {
		for (j=col_start;j<(col_start+dim);j++) {
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

void parallel_lu(int argc, char **argv, double **matrix, int dim, int block_dim, int rank2print, int doSerial, int numThreads) {
	omp_set_num_threads(numThreads);
  int procs;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	MPI_Request request;

	int num_rows = sqrt(procs);
	int num_cols = sqrt(procs);

	int dimSize[2] = {num_rows, num_cols};
	int periodic[2] = {0, 0};
	int myCoords[2];

	MPI_Comm comm2D;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, periodic, 0, &comm2D);

	int myRow, myCol;
	MPI_Cart_coords(comm2D, rank, 2, myCoords);
	myRow = myCoords[0]; 
	myCol = myCoords[1];

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

	int i,j,k;
	// initialize buffers
	for (i=0;i<block_dim;i++) {
		LBuffSend[i] = LBuffRecv[i] = PBuffSend[i] = PBuffRecv[i] = 0;
	} 

	// initialize L diag
	for (i=0;i<dim;i++) {
		L[i][i] = 1.0;
	}

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
		//print_matrix(dim,matrix);
	}

	//Main computation loop
	for(k=0;k<dim;k++) {
		bool kInMyRows = k >= row_start && k <= row_end;
		bool kInTopRows = k <= row_end-block_dim;
		bool kInBotRows = k >= row_start+block_dim;
		
		bool kInMyCols = k>=col_start && k<=col_end;
		bool kInLeftCols = k <= col_end-block_dim;
		bool kInRightCols = k >= col_start+block_dim;

		//Send & recieve pivot row
		//Recieve PBuffRec from top
		if(topRank >= 0 && kInTopRows && !kInRightCols) {
			MPI_Recv(PBuffRecv, block_dim, MPI_DOUBLE, topRank, 0, MPI_COMM_WORLD, &status);
			if(rank==rank2print) {
				printf("Received pivot row from rank %i for k = %i: ",topRank,k);
				print_vector(block_dim,PBuffRecv);
			}
			//Place PBuffRecv in correct place of matrix
			for(j=col_start;j<=col_end;j++) {
				if(j>=k) {
					matrix[k][j] = PBuffRecv[j-col_start];
				}
			}
		}
		//send PBuffSend to bottom 
		if(botRank >= 0 && !kInRightCols) {
			if(kInMyRows) { //pivot row is generated from this process
				//Assemble PBuffSend
				for(j=col_start;j<=col_end;j++) {
					if(j>=k) {
						PBuffSend[j-col_start] = matrix[k][j];
					}
				}
				if(rank==rank2print) {
					printf("Sending pivot row to rank %i for k = %i (Creating): ",botRank,k);
					print_vector(block_dim,PBuffSend);
				}
			}
			else if(kInTopRows) { //pivot row is generated in a top process; just pass the recieved one along
				//Assemble PBuffSend
				for(j=col_start;j<=col_end;j++) {
					if(j>=k) {
						PBuffSend[j-col_start] = PBuffRecv[j-col_start];
					}
				}
				if(rank==rank2print) {
					printf("Sending pivot row to rank %i for k = %i (Passing): ",botRank,k);
					print_vector(block_dim,PBuffSend);
				}
			}
			MPI_Isend(PBuffSend, block_dim, MPI_DOUBLE, botRank, 0, MPI_COMM_WORLD, &request);
		} 

		//Calculate ratios
		
		if(kInMyCols) {
			for(i=row_start;i<=row_end;i++) {
				if (i>k) {
					L[i][k] = matrix[i][k]/matrix[k][k];
				}
			}
		}

		//Wait for PBuffSend to be usable
		if(botRank >= 0 && kInMyRows)
			MPI_Wait(&request, &status);

		if(rank==rank2print) {
			printf("L:\n");
			print_matrix_chunk(block_dim,row_start,col_start,L);
		}

		//Send & recieve ratios
		//Recieve LBuffRec from left
		if(leftRank >= 0 && kInLeftCols && !kInBotRows) {
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
		//send LBuffSend to right
		if(rightRank >= 0 && !kInBotRows) { 
			if(kInMyCols) {  //ratio is generated from this process
				//Assemble LBuffSend
				for(i=row_start;i<=row_end;i++) {
					if(i>k) {
						LBuffSend[i-row_start] = L[i][k];
					}
				}
				if(rank==rank2print) {
					printf("Sending L to rank %i for k = %i: (Creating)",rightRank,k);
					print_vector(block_dim,LBuffSend);
				}
			}
			else if(kInLeftCols) { //ratio is generated in a left process; just pass the recieved one along
				//Assemble LBuffSend
				for(i=row_start;i<=row_end;i++) {
					if(i>k) {
						LBuffSend[i-row_start] = LBuffRecv[i-row_start];
					}
				}
				if(rank==rank2print) {
					printf("Sending L to rank %i for k = %i (Passing): ",rightRank,k);
					print_vector(block_dim,LBuffSend);
				}
			}
			MPI_Isend(LBuffSend, block_dim, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, &request);
		}

		//Compute upper triangular matrix
    #pragma omp parallel for private(j,i) firstprivate(k,col_start,col_end) 
		for (j=col_start;j<=col_end;j++) {
			if (j>=k) {
				for (i=row_start;i<=row_end;i++) {
					if (i>k) {
						matrix[i][j] = matrix[i][j]-L[i][k]*matrix[k][j];
					}
				}
			}
		}

		//Wait for LBuffSend to be usable
		if(rightRank >= 0 && kInMyCols)
			MPI_Wait(&request, &status);

		if(rank==rank2print) {
			printf("U:\n");
			print_matrix_chunk(block_dim,row_start,col_start,matrix);
		}

	}

	/*
	double **L_chunk = create_zero_matrix(block_dim);
	double **U_chunk = create_zero_matrix(block_dim);
	// copy chunk data
	int r = 0;
	for(i=row_start;i<=row_end;i++) {
	int c = 0;
	for(j=col_start;j<=col_end;j++) {
	L_chunk[r][c] = L[i][j];
	U_chunk[r][c] = matrix[i][j];  
	c++;
	}
	r++;
	}*/

	if(rank2print == -1) {
		printf("Rank %i\n",rank);
		printf("L\n"); 
		print_matrix_chunk(block_dim,row_start,col_start,L);
		//print_matrix(block_dim,L_chunk);
		printf("U\n"); 
		print_matrix_chunk(block_dim,row_start,col_start,matrix);
		//print_matrix(block_dim,U_chunk);
	}

	/*if(rank != 0) {
	// send L and U chunks to process 0
	MPI_Isend(L_chunk,block_dim*block_dim,MPI_DOUBLE,0,rank*,MPI_COMM_WORLD,&request);
	} else {
	// receive L and U chunks from all processes
	}*/

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
	int procs;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double **L = create_zero_matrix(dim);

	if(rank==0) {
		printf("A=\n");
		print_matrix(dim,matrix);
	}

	int i,j,k;
	// initialize diagonal of L to 1
	for(i=0;i<dim;i++) {
		L[i][i] = 1;
	}

	// columns
	for(j=0;j<dim;j++) {
		// rows
		for(i=j+1;i<dim;i++) {
			double ratio = matrix[i][j]/matrix[j][j];
			L[i][j] = ratio;
			for(k=0;k<dim;k++) {
				matrix[i][k] -= ratio*matrix[j][k];
			}
		}
	}

	//Print serial results for proc 0 only (computation done on all)
	if(rank == 0) {
		printf("L=\n");
		print_matrix(dim,L);
		printf("U=\n");
		print_matrix(dim,matrix);
		printf("L*U=\n");
		print_matrix(dim,matrix_mult(L,matrix,dim));
	}
	free_matrix(dim,L);
	free_matrix(dim,matrix);
}


int main(int argc, char **argv) {
	//int dims[3] = {5,10,15};
	//int dim = 4;
	//int block_dim = 2;
  double time,avg;
  MPI_Init(&argc, &argv);
	int procs;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if doPAPI==1
	data = 0;
	papi_setup();
#endif

	//Check if number of processes is a perfect square
	if (sqrt(procs) != floor(sqrt(procs))) {
		printf("Number of processes is not a perfect square!\n");
		return -1;
	}

	//Read input arguements
	if(argc < 2) {
		if(rank == 0)
			fprintf(stderr,"Wrong # of arguments.\nUsage: mpirun -n procs %s $dim $numThreads $rank2print $doSerial(Only dim and numThreads are required; other 2 are optional)\n",argv[0]);
		return -1;
	}
	int dim = atoi(argv[1]);
  int numThreads = atoi(argv[2]);
	int block_dim = dim/(sqrt(procs));
	int rank2print = -1;
	int doSerial = 0;
	if(argc == 4) {
		rank2print = atoi(argv[3]);
	}
	if(argc == 5) {
		rank2print = atoi(argv[3]); 
		doSerial = atoi(argv[4]);
	}

	//Run code
	if(rank==0)
		printf("Running code on %i procs with dim = %i; numThreads = %i; block_dim = %i; printing on rank %i; doSerial = %i \n",procs, dim, numThreads, block_dim, rank2print, doSerial);
	if(doSerial==1) 
		serial_lu(generate_matrix(dim),dim);
#if doPAPI==1
	papi_start();
	parallel_lu(argc, argv, generate_matrix(dim), dim, block_dim, rank2print, doSerial, numThreads);
	papi_report();
#else
  time = MPI_Wtime();
  parallel_lu(argc, argv, generate_matrix(dim), dim, block_dim, rank2print, doSerial, numThreads);
  time = MPI_Wtime() - time;
  MPI_Reduce(&time, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank == 0) {
    printf("Dim = %i, Procs = %i, Threads = %i, Average time: %e\n", dim, procs, numThreads, avg/procs);
  }
#endif
  MPI_Finalize();
	return 0;
}
