nvcc oddeven.cu
nvcc bitinic.cu
nvcc bucket.cu
mpicc -o oddeven.bin ./oddeven.c
mpicc -o bitonic.bin ./bitonic.c
mpicc -o bucket.bin ./bucket.c
mpirun -np 5 ./oddeven.bin 1000
mpirun -np 5 ./bitonic.bin 1000
mpirun -np 5 ./bucket.bin 1000
