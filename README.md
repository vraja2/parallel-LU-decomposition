# parallel-LU-decomposition
CS 420 Final Project: Parallel LU Decomposition


Added input arguments for running code:
Usage is now: mpirun -n $procs ./parallel_decomp $dim $numThreads $rank2print $doSerial (Only dim and numThreads are required; other 2 are optional). Set $rank2print as -1 for global print. Defaults are rank2print = -1 and doSerial = 0

Example: mpirun -n 4 ./parallel_decomp 4 1 -1 1
