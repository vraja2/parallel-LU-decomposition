# parallel-LU-decomposition
CS 420 Final Project: Parallel LU Decomposition


Running the code: 
Block mapping: mpirun -n $procs ./parallel_decomp $dim $numThreads $rank2print $doSerial (Only dim and numThreads are required; other 2 are optional). Set $rank2print as -2 for quiet output. Defaults are rank2print = -1 and doSerial = 0

Example: mpirun -n 4 ./parallel_decomp 4 1 -1 1

Cyclic mapping: mpirun -n $procs ./parallel_decomp_cyclic $dim $numThreads $block_dim $rank2print $doSerial (Only dim, numThreads and block_dim are required; other 2 are optional). Set $rank2print as -2 for quiet output. Defaults are rank2print = -1 and doSerial = 0
