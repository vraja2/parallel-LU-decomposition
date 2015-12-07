#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define printdebug 0


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

void print_matrix_chunk_cyclic(int dim, int blocksInRank, int* row_start, int* col_start, double** matrix) {
	// Only for square matrix chunks
	int i,ii,j,jj;
	for(ii=0;ii<blocksInRank;ii++) {
		for(i=row_start[ii];i<row_start[ii]+dim;i++) {
			for(jj=0;jj<blocksInRank;jj++) {
				for (j=col_start[jj];j<col_start[jj]+dim;j++) {
					if (abs(matrix[i][j]) < 10) {
						printf("%f     ",matrix[i][j]);
					} else if (abs(matrix[i][j]) < 100) {
						printf("%f    ", matrix[i][j]);
					} else {
						printf("%f  ", matrix[i][j]);
					}
				}
			}
			printf("\n");
		}
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

void parallel_lu_cyclic(int argc, char **argv, double **matrix, int dim, int block_dim, int rank2print, int doSerial, int numThreads) {
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
	int periodic[2] = {1, 1};
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

	int sqrprocs = sqrt(procs);
	int numsInRank = dim/sqrprocs;
	int blocksInRank = numsInRank/block_dim;
	
	double **L = create_zero_matrix(dim);
	double *LBuffSend = (double*) malloc (numsInRank * sizeof(double));
	double *LBuffRecv = (double*) malloc (numsInRank * sizeof(double));
	double *PBuffSend = (double*) malloc (numsInRank * sizeof(double));
	double *PBuffRecv = (double*) malloc (numsInRank * sizeof(double));
	int LBuffCtr, PBuffCtr;
	int LSendCtr, PSendCtr; //Counters to ensure each process gets new data only once
	int sendMax = procs;

	int i,ii,j,jj,k;
	// initialize buffers
	for (i=0;i<numsInRank;i++) {
		LBuffSend[i] = LBuffRecv[i] = PBuffSend[i] = PBuffRecv[i] = 0;
	} 

	// initialize L diag
	for (i=0;i<dim;i++) {
		L[i][i] = 1.0;
	}

	int proc_per_row = dim/block_dim; 
	/*int col_start = (rank*block_dim) % dim;
	int col_end = col_start+block_dim-1;
	int row_start = (rank/proc_per_row)*block_dim;
	int row_end = row_start+block_dim-1;*/

	int col_start = myCol*block_dim;
	int col_end = col_start+block_dim-1;
	int row_start = myRow*block_dim;
	int row_end = row_start+block_dim-1;

	int rowsInRank[blocksInRank];
	int colsInRank[blocksInRank];
	for(i=0;i<blocksInRank;i++) {
		rowsInRank[i] = row_start + i*block_dim*sqrprocs;
		colsInRank[i] = col_start + i*block_dim*sqrprocs;
	}

	bool atTopBound[blocksInRank]; bool atBotBound[blocksInRank];
	bool atRightBound[blocksInRank]; bool atLeftBound[blocksInRank];

	for(i=0; i<blocksInRank; i++) {
		atTopBound[i] = rowsInRank[i] == 0;
		atBotBound[i] = rowsInRank[i]+block_dim == dim;
		atLeftBound[i] = colsInRank[i] == 0;
		atRightBound[i] = colsInRank[i]+block_dim == dim;
	}


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
		printf("rowsInRank : ");
		for(i=0;i<blocksInRank;i++) {
			printf("%i ",rowsInRank[i]);
		}
		printf("\n");
		printf("colsInRank : ");
		for(i=0;i<blocksInRank;i++) {
			printf("%i ",colsInRank[i]);
		}
		printf("\n");
		printf("atTopBound : ");
		for(i=0;i<blocksInRank;i++) {
			printf("%i ",atTopBound[i]);
		}
		printf("\n");
		printf("atBotBound : ");
		for(i=0;i<blocksInRank;i++) {
			printf("%i ",atBotBound[i]);
		}
		printf("\n");
		printf("atLeftBound : ");
		for(i=0;i<blocksInRank;i++) {
			printf("%i ",atLeftBound[i]);
		}
		printf("\n");
		printf("atTopBound : ");
		for(i=0;i<blocksInRank;i++) {
			printf("%i ",atLeftBound[i]);
		}
		printf("\n");
		printf("Original matrix on this process is: \n");
		print_matrix_chunk_cyclic(block_dim, blocksInRank, rowsInRank, colsInRank, matrix);
	}



	//Main computation loop
	for(k=0;k<dim;k++) {
		bool kInMyRows[blocksInRank]; bool kInTopRows[blocksInRank]; bool kInBotRows[blocksInRank];
		bool kInMyCols[blocksInRank]; bool kInTopCols[blocksInRank]; bool kInLeftCols[blocksInRank];

		for(i=0; i<blocksInRank; i++) {
			kInMyRows[i] = k >= rowsInRank[i] && k <= rowsInRank[i]+block_dim-1;
			kInTopRows[i] = k < rowsInRank[i];
			kInMyCols[i] = k >= colsInRank[i] && k <= colsInRank[i]+block_dim-1;
			kInLeftCols[i] = k < colsInRank[i];

			if(rank==rank2print && printdebug==1) {
				printf("kInMyRows for k = %i and row start at %i is: %i\n",k,rowsInRank[i],kInMyRows[i]);
				printf("kInMyTopRows for k = %i and row start at %i is: %i\n",k,rowsInRank[i],kInTopRows[i]);
				//printf("kInMyCols for k = %i and col start at %i is: %i\n",k,colsInRank[i],kInMyCols[i]);
			}
		}
		
		//Send & recieve pivot row
		PSendCtr = 0;
		for(i=0; i<blocksInRank; i++) {
			//Recieve PBuffRec from top
			if(!atTopBound[i] && kInTopRows[i] && PSendCtr<=sendMax) {
				PBuffCtr = 0;
				MPI_Recv(PBuffRecv, numsInRank, MPI_DOUBLE, topRank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				PSendCtr = status.MPI_TAG;
				if(rank==rank2print) {
					printf("Received pivot row from rank %i for k = %i and i = %i: ",topRank,k,i);
					print_vector(numsInRank,PBuffRecv);
					printf("PSendCtr is: %i\n",PSendCtr);
				}
				//Place PBuffRecv in correct place of matrix 
				for(jj = 0; jj<blocksInRank; jj++) {
					for(j=colsInRank[jj];j<colsInRank[jj]+block_dim;j++) {
						if(j>=k) {
							matrix[k][j] = PBuffRecv[PBuffCtr];
						}
						PBuffCtr++;
					}
				}
			}
			//send PBuffSend to bottom 
			if(!atBotBound[i] && PSendCtr<=sendMax) {
				if(kInMyRows[i]) { //pivot row is generated from this process
					PBuffCtr = 0;
					++PSendCtr;
					//Assemble PBuffSend
					for(jj = 0; jj<blocksInRank; jj++){
						for(j=colsInRank[jj];j<colsInRank[jj]+block_dim;j++) {
							if(j>=k) {
								PBuffSend[PBuffCtr] = matrix[k][j];
							}
							PBuffCtr++;
						}
					}
					if(rank==rank2print) {
						printf("Sending pivot row to rank %i for k = %i and i = %i(Creating): ",botRank,k,i);
						print_vector(numsInRank,PBuffSend);
						printf("PSendCtr is: %i\n",PSendCtr);
					}
					MPI_Send(PBuffSend, numsInRank, MPI_DOUBLE, botRank, PSendCtr, MPI_COMM_WORLD);
				}
				else if(kInTopRows[i]) { //pivot row is generated in a top process; just pass the recieved one along
					PBuffCtr = 0;
					++PSendCtr;
					//Assemble PBuffSend
					for(jj = 0; jj<blocksInRank; jj++){
						for(j=colsInRank[jj];j<colsInRank[jj]+block_dim;j++) {
							if(j>=k) {
								PBuffSend[PBuffCtr] = PBuffRecv[PBuffCtr];
							}
							PBuffCtr++;
						}
					}
					if(rank==rank2print) {
						printf("Sending pivot row to rank %i for k = %i (Passing) and i = %i: ",botRank,k,i);
						print_vector(numsInRank,PBuffSend);
						printf("PSendCtr is: %i\n",PSendCtr);
					}
					MPI_Send(PBuffSend, numsInRank, MPI_DOUBLE, botRank, PSendCtr, MPI_COMM_WORLD);
				}
				//MPI_Isend(PBuffSend, numsInRank, MPI_DOUBLE, botRank, 0, MPI_COMM_WORLD, &request);
			} 
		}
		
		
		//Calculate ratios
		
		for(j=0; j<blocksInRank; j++) {
			if(kInMyCols[j]) {
				for(ii=0; ii<blocksInRank;ii++){
					for(i=rowsInRank[ii];i<rowsInRank[ii]+block_dim;i++) {
						if (i>k) {
							L[i][k] = matrix[i][k]/matrix[k][k];
							//if(rank==rank2print)
								//printf("Computing ratio for row = %i and col = %i\n",i,k);
						}
					}
				}
			}
		}

		//Wait for PBuffSend to be usable
		//MPI_Wait(&request, &status);		
	
		if(rank==rank2print) {
			printf("L:\n");
			print_matrix_chunk(block_dim,rowsInRank[0],colsInRank[blocksInRank-1],L);
		}

		//Send & recieve ratios
		LSendCtr=0;
		for(j=0; j<blocksInRank; j++) {
			//Recieve LBuffRec from left
			if(!atLeftBound[j] && kInLeftCols[j] && LSendCtr<=sendMax) {
				LBuffCtr=0;
				MPI_Recv(LBuffRecv, numsInRank, MPI_DOUBLE, leftRank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				LSendCtr = status.MPI_TAG;
				if(rank==rank2print) {
					printf("Recieved L from rank %i for k = %i: ",leftRank,k);
					print_vector(numsInRank,LBuffRecv);
					printf("LSendCtr is: %i\n",LSendCtr);
				}
				//Place LBuffRecv in correct place of L[i][k]
				for(ii=0;ii<blocksInRank;ii++) {
					for(i=rowsInRank[ii];i<rowsInRank[ii]+block_dim;i++) {
						if(i>k) {
							L[i][k] = LBuffRecv[LBuffCtr];
						}
						LBuffCtr++;
					}
				}
			}
			//send LBuffSend to right
			if(!atRightBound[j] && LSendCtr<=sendMax) { 
				if(kInMyCols[j]) {  //ratio is generated from this process
					LBuffCtr = 0;
					++LSendCtr;
					//Assemble LBuffSend
					for(ii=0;ii<blocksInRank;ii++) {
						for(i=rowsInRank[ii];i<rowsInRank[ii]+block_dim;i++) {
							if(i>k) {
								LBuffSend[LBuffCtr] = L[i][k];
							}
							LBuffCtr++;
						}
					}
					if(rank==rank2print) {
						printf("Sending L to rank %i for k = %i: (Creating)",rightRank,k);
						print_vector(numsInRank,LBuffSend);
						printf("LSendCtr is: %i\n",LSendCtr);
					}
					MPI_Send(LBuffSend, numsInRank, MPI_DOUBLE, rightRank, LSendCtr, MPI_COMM_WORLD);
				}
				else if(kInLeftCols[j]) { //ratio is generated in a left process; just pass the recieved one along
					LBuffCtr = 0;
					++LSendCtr;
					//Assemble LBuffSend
					for(ii=0;ii<blocksInRank;ii++) {
						for(i=rowsInRank[ii];i<rowsInRank[ii]+block_dim;i++) {
							if(i>k) {
								LBuffSend[LBuffCtr] = LBuffRecv[LBuffCtr];
							}
							LBuffCtr++;
						}
					}
					if(rank==rank2print) {
						printf("Sending L to rank %i for k = %i (Passing): ",rightRank,k);
						print_vector(numsInRank,LBuffSend);
						printf("LSendCtr is: %i\n",LSendCtr);
					}
					MPI_Send(LBuffSend, numsInRank, MPI_DOUBLE, rightRank, LSendCtr, MPI_COMM_WORLD);
				}
			}
		}

		//Compute upper triangular matrix
		#pragma omp parallel for private(j,i,ii,jj) firstprivate(k) 
		for(jj = 0; jj<blocksInRank;jj++) {
			for (j=colsInRank[jj];j<colsInRank[jj]+block_dim;j++) {
				if (j>=k) {
					for(ii = 0; ii<blocksInRank; ii++) {
						for (i=rowsInRank[ii];i<rowsInRank[ii]+block_dim;i++) {
							if (i>k) {
								matrix[i][j] = matrix[i][j]-L[i][k]*matrix[k][j];
							}
						}
					}
				}
			}
		}

		//Wait for LBuffSend to be usable
		//MPI_Wait(&request, &status);

		if(rank==rank2print) {
			printf("U:\n");
			print_matrix_chunk(block_dim,rowsInRank[0],colsInRank[blocksInRank-1],matrix);
		}
	}
	
	if(rank==rank2print) {
		printf("U:\n");
		print_matrix_chunk_cyclic(block_dim, blocksInRank, rowsInRank, colsInRank, matrix);
	}

	if(rank==rank2print) {
		printf("L:\n");
		print_matrix_chunk_cyclic(block_dim, blocksInRank, rowsInRank, colsInRank, L);
	}


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
	double time,avg;
	MPI_Init(&argc, &argv);
	int procs;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Check if number of processes is a perfect square
	if (sqrt(procs) != floor(sqrt(procs))) {
		printf("Number of processes is not a perfect square!\n");
		return -1;
	}

	//Read input arguements
	if(argc < 4) {
		if(rank == 0)
			fprintf(stderr,"Wrong # of arguments.\nUsage: mpirun -n $procs %s $dim $numThreads $block_dim $rank2print $doSerial (Only dim, numThreads & width is required; other 2 are optional)\n",argv[0]);
		return -1;
	}
	int dim = atoi(argv[1]);
	int numThreads = atoi(argv[2]);
	int block_dim = atoi(argv[3]);
	//int block_dim = dim/(sqrt(procs));
	int rank2print = -1;
	int doSerial = 0;
	if(argc == 5) {
		rank2print = atoi(argv[4]);
	}
	if(argc == 6) {
		rank2print = atoi(argv[4]); 
		doSerial = atoi(argv[5]);
	}

	//Run code
	if(rank==0)
		printf("Running cyclic code on %i procs with dim = %i; numThreads = %i; block_dim = %i; printing on rank %i; doSerial = %i \n",procs, dim, numThreads, block_dim, rank2print, doSerial);
	if(doSerial==1) 
		serial_lu(generate_matrix(dim),dim);
	time = MPI_Wtime();
	parallel_lu_cyclic(argc, argv, generate_matrix(dim), dim, block_dim, rank2print, doSerial, numThreads);
	 time = MPI_Wtime() - time;
	MPI_Reduce(&time, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Dim = %i, Block Dim = %i, Procs = %i, Threads = %i, Average time: %e\n", dim, block_dim, procs, numThreads, avg/procs);
	}
	MPI_Finalize();
	return 0;
}
