/*
������������ ����������(bitonic sort)
� ������ ���� ���������� ����� �������� Bn(��������������, half - cleaner) ��� ��������, �����������
��������������� �������� ��� xi � xi + n / 2.�� ���. 1 �������������� ����� ������������� �������� ��� ��� ��
�����������, ��� � �� ��������.���������� �������� �� ������� ������������ ������������������ �
����������� : ���� ����� ��������������� ��������� ��������� ������������ ������������������ ����� �
			  ������, �� �� ��������� ��������� ������������ ������������������.
			  ������������������ a0, a1, �, an - 1 ���������� ������������, ���� ��� ��� ������� �� ���� ����������
			  ������(�.�.���� ������� ����������, � ����� �������, ���� ��������), ��� �������� ����� ������������
			  ������ �� ����� ������������������.���, ������������������ 5, 7, 6, 4, 2, 1, 3 ������������, ���������
			  �������� �� 1, 3, 5, 7, 6, 4, 2 ����� ������������ ������ ����� �� ��� ��������.
			  ��������, ��� ���� ��������� �������������� Bn � ������������ ������������������ a0, a1, �, an - 1,
			  �� ������������ ������������������ �������� ���������� ���������� :
� ��� �� �������� ����� ����� �������������.
� ����� ������� ������ �������� ����� �� ������ ������ �������� ������ ��������.
� ���� �� ���� �� ������� �������� ����������.
�������� � ������������ ������������������ a0, a1, �, an - 1 �������������� Bn, ������� ���
������������������ ������ n / 2, ������ �� ������� ����� ������������, � ������ ������� ������ �� ��������
������ ������� ������.����� �������� � ������ �� ������������ ������� �������������� Bn / 2.�������
��� ������ ������������ ������������������ ����� n / 4.�������� � ������ �� ��� �������������� Bn / 2 �
��������� ���� ������� �� ��� ���, ���� �� ������ � n / 2 ������������������� �� ���� ���������.�������� �
������ �� ��� �������������� B2, ����������� ��� ������������������.��������� ��� ������������������
��� �����������, ��, ��������� ��, ������� ��������������� ������������������.
����, ���������������� ���������� ��������������� Bn, Bn / 2, �, B2 ��������� ������������
������������ ������������������.��� �������� �������� ������������ �������� � ���������� Mn.
��������, � ������������������ �� 8 ��������� a 0, a1, �, a7 �������� �������������� B2, ����� ��
�������� ����� ������� ���������� ��� ��������������.�� ���. 2 �����, ��� ������ ������ ��������
������������ ������������������ �������� ������������ ������������������.���������� ���������
������ �������� ����� �������� ������������ ������������������.������� ������ �� ���� ������� �����
������������� ������������ ��������, ������ �������� ������� ����� �������, ����� �����������
���������� � ��������� ���� ���������������.� ���������� ��� �������� �������� ������ ������������
������������ ���������� ������������������ �� n ��������� ����������� ������� � ������ ��
������� ����������� � ����� �����������.����� ����� ���������� ������������ ������������������
����������� ������������ ��������.
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef byte char;
typedef size_t int;
typedef __device__ int(*comparer)(byte *x, byte *y, size_t size);
typedef __device__ int(*indexator)(byte *x, int index, int len, size_t size);
typedef __device__ void(*non_parallel_sort)(byte *arr, int index, int len, int n, size_t size, int direction);
typedef __host__ void(*parallel_sort)(byte *data, int n, size_t size, int direction);

////////////////////////////////////////////////////////////////////////////////////////////
// ����������� ���������
// _comparer - ������� ��������� ���� ��������� �������
// _indexator - ������� ����������� ������ ������� ��� �������� �������
// _non_parallel_sort - ������ ���������� ��� ������������� ����������� ����������
// _parallel_sort - ������ ���������� � �������������� ����������� ����������

comparer _comparer;
indexator _indexator;
non_parallel_sort _non_parallel_sort;
parallel_sort _parallel_sort;

__host__ void host_bitonic_sort(byte *data, int n, size_t size, int direction)
{
// data - ������ ������
// n - ���������� ��������� � �������� ������� ��� ����������
// size - ������ ������ �������� ������� � ������
// direction - ������ ���������� 
// -1 �������� ���������� �� ��������, 
//  1 �������� ���������� �� �����������
	
	cudaError_t err;
	byte *device_data;

	// ����� ���� ��������� k*(k-1)/2*2^(k-1) �������� ���������, ��� k = log2 n
	// �� ���� �������� ������� ��������� ����� ��������� n/2 = 2^(k-1) ��������

	// ��������� ����������� ��������� �� ��������, ���� � ����� 
	// ���� ���� � �������� ����� ����� ��������� ����� � ��������� ����������� ��������

	int blocks = min(max(1,(int)pow((double)n,0.33333333333)),255);
	int threads = max(1,(int)sqrt((double)n/blocks));
	int loops = (int)(n+2*blocks*threads-1)/(2*blocks*threads);

	assert(n <= 2*blocks*threads*loops);

	// ��� ������ - �������� �������� ������ � ������ GPU 

	error = cudaMalloc((void**)&device_data, n*size);
	cudaMemcpy(device_data, data, n*size, cudaMemcpyHostToDevice);

	int i = 0;
	do {
		i++; // �������� - 1 ������� �����
		for( int j = i; j-- > 0 ; ) // �������� ������� ����
		{ 
			// ���������� ��� � ������ ����� ����������� ���������� �������� (�������������� ������� � ����� � ��� �� ������)
			global_bitonic_worker <<< blocks, threads >>>(device_data, n, i, j, loops, size, direction);
		}
	}
	while( (1<<i) < n );

	// ���������� ���������� � �������� ������
	cudaMemcpy(data, device_data, n*size, cudaMemcpyDeviceToHost);

	// ����������� ������ �� ����������
	cudaFree(device_data);
}

__global__ void global_bitonic_worker(
	bype * data, 
	int n, int i, int j,
	int loops,
	size_t size,
	int direction)
{
	// �������� ������������� ����
	int block = blockDim.x*blockIdx.x + threadIdx.x;
	for(int y=0; y<loops; y++) {
		// �������� ������������� ���� �����
		int id = block*loops+y;
		int step = 1<<j;
		int offset = ((id>>j)<<(j+1))+(id&((1<<j)-1);
		if ((offset+step) < n) {
			int parity = (id>>i);
			while(parity>1) parity = (parity>>1) ^ (parity&1);
			parity = (parity<<1)-1; // ������ ���������� parity ����� ����� ������ 2 �������� 1 � -1
			int value = parity*direction*(comparer)(&data[offset*size],&data[(offset+step)*size],size);
			if (value < 0) device_exchange(&data[index*size],&data[(index+step)*size],size);
		}
	}
}

// ������������ ���� ������ � ������ ����������
__device__ void device_exchange(byte *x, byte *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		byte ch = x[i] ; x[i] = y[i] ; y[i] = ch;
	}
}

// ����������� ������ ������� ������ � ������
__device__ void device_copy(byte *x, byte *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		x[i] = y[i] ;
	}
}

// ������� ��������� ������ x������� � ������ ��� ����� ����� ���� long
__device__ int device_integer_comparer(byte *x, byte *y, size_t size)
{
	assert(size == sizeof(long));
	if ((*(long*)x)<(*(long*)y)) return 1;
	else if ((*(long*)x)>(*(long*)y)) return -1;
	else return 0;
}

// ����������� ������ ������� 
// ����������� ������������� ����� �� len ��� � ������� index
__device__ int device_integer_indexator(byte *x, int index, int len, size_t size)
{
	assert(size == sizeof(long));
	return min(m, (((*(long*)x) >> index) + (1 << (8 * sizeof(long)-index))) ^ ((1 << len) - 1);
}

/////////////////////////////////////////////////////////////////
// ������������ ����������
__device__ void device_bitonic_sort(byte *arr, int index, int len, int n, size_t size, int direction)
{
	assert(index+len < n);
	assert(len > 0);
	int i = 0;
	do {
		i++; // �������� - 1 ������� ����� 
		for( int j = i; j-- > 0 ; ) // �������� ������� ����
		{ 
			for(int id=0; (2*id) < len; id++)
			{
				int step = 1<<j;
				int offset = ((id>>j)<<(j+1))+(id&((1<<j)-1);
				if ((offset+step) < len) {
					int parity = (id>>i);
					while(parity>1) parity = (parity>>1) ^ (parity&1);
					parity = (parity<<1)-1; // ������ ���������� parity ����� ����� ������ 2 �������� 1 � -1
					int value = parity*direction*(comparer)(&data[offset*size],&data[((offset+step)%n)*size],size);
					if (value < 0) device_exchange(&data[offset*size],&data[((offset+step)%n)*size],size);
				}
			}
		}
	}
	while( (1<<i) < len );
}

int main(int argc, char* argv[])
{
	// Find/set the device.
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
	}

	_comparer = device_integer_comparer;
	_indexator = device_integer_indexator;
	_non_parallel_sort = device_bitonic_sort;
	_parallel_sort = host_bitonic_sort;

	for (int n = 10; n < 10000; n *= 10)
	{
		// ������ ������ ����� n ����� ���� long
		// ��������� ������ ������-���������� ���������� ��������� ������� rand

		long *arr = (long *)malloc(n*sizeof(long));
		for (int i = 0; i<n; i++) { arr[i] = rand(); }

		// ��������� ������ �� �����������
		time_t start = time(NULL);
		(*_parallel_sort)(arr, n, sizeof(long), 1);
		time_t end = time(NULL);

		// ���������
		bool check = true;
		for (int i = 0; (i < (n - 1)) && check; i++)
			check = (arr[i] <= arr[i + 1]);

		std::cout << "n = " << n << "\t" << "time = " << (end - start) << "\t" << "results :" << (check ? "ok" : "fail") << std::endl;

		// ������������ ������
		free(arr);
	}

	cudaDeviceReset();

	exit(0);
}