/*
���������� �����-��������� �������������� (odd-even sort)
��� ������ �������� ��������� �������� ���������-������ ��� ���� ��� ��������� ���������� �
����������� ������������. ���������� ������, ����� ����� ����������� ����� ����� ���������, �.�. p=n -
����� ����������� (����������� ���������). �����������, ��� �������������� ������� ����� ���������
������. ����� �������� ai (i = 1, .. , n), ������������� ����������� �� ����������� pi (i = 1, ... , n). � ��������
�������� ������ ��������� � �������� ������� ���������� ���������-����� ������ �������� � ���������,
����������� �� ����������-������ ������. ���������� � ������� ������ �������� ������ ��������� � ������
������� ���������� ���������-����� ������ �������� � ��������� ������� ������.
�� ������ �������� ��������� �������� � ������ ���������� ��������� ��� ���������-������ � ��
������� �������� �� ����� Q(1). ����� ���������� ����� �������� � n; ������� ����� ����������
������������ ���������� � Q(n).
����� ����� ����������� p ������ ����� ��������� n, �� ������ �� ��������� �������� ���� ����
������ n/p � ��������� ��� �� ����� Q((n/p)�log(n/p)). ����� ���������� �������� p �������� (�/2 � ������, �
��������) � ������ �����������-���������: ������� ���������� �������� ���� ����� ���� ������, �
��������� �� ��������� (�� ������ ���� ����������� �������� ���������� �������). ����� ���������
������ ������� �� 2 �����; ����� ��������� ������������ ����� ������ ����� ����� (� �������� ����������
������), � ������ � ������ ������ (� �������� ���������� ������). �������� ��������������� ������
����� p ��������, �������� � ������ ����� ����:
23)���� ��������� ������� ������ �� ���������;
24)���� ������������ ������ ����� �������;
25)����� ��������� ���� ����� ������;
26)����� ������������ �������, ��� ���� ��������� ������ � �������� �����;
27)���� ���������� ������������ ����� ���������� ������, �� �������� ��������;
28)���� ��������� ��������������� ������.

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

__host__ void host_oddeven_sort(byte *data, int n, size_t size, int direction)
{
// data - ������ ������
// n - ���������� ��������� � �������� ������� ��� ����������
// size - ������ ������ �������� ������� � ������
// direction - ������ ���������� 
// -1 �������� ���������� �� ��������, 
//  1 �������� ���������� �� �����������
	
	cudaError_t err;
	byte *device_data;
	int * device_index;

	// ����������� ������������ ��������� �� �����, �������� � ����
	// ���� ���� � �������� ����� ����������� ������ ����� 2*block_length

	int block_length = (int)((max(1,(int)pow((double)n,0.33333333333))+1)/2);

	// ��� ���������� ��������� ��� ����������� ������ ��������� �� ��� ������, 
	// �� ���� ���������� ������ ������ ���� ������ 2 
	// ��������� �� ����� ���� ���������� ��� ����� ����������� �� ���� 
	
	int number_block_pairs = (int)((n+2*block_length-1)/(2*block_length));

	// ��������� ����������� ��������� �� ��������, ����
	// ���� ���� � �������� ����� ����������� ������ ����� 2*block_length

	int blocks = max(1,(int)sqrt((double)number_block_pairs))),255);
	int threads = (int)((number_block_pairs+number_tasks-1)/number_tasks);

	assert(n <= 2*block_length*number_block_pairs);
	assert(number_block_pairs <= number_tasks*number_threads);
	
	// ��� ������ - �������� �������� ������ � ������ GPU 

	error = cudaMalloc((void**)&device_data, n*size);
	cudaMemcpy(device_data, data, n*size, cudaMemcpyHostToDevice);

	// ��� ������ - �������� ��� �������� ������� ����� �������
	// �������������� ��� ����� ��������� ������ ������ �����, 
	// � ��� ����������� 1 ��������������� ������
	// � ����������� ��������� � ���������� ������

	error = cudaMalloc((void**)&device_index, (2*number_block_pairs+1)*sizeof(int));

	// ���������� ����������� �������� ������ ����� �������
	device_index[2*number_block_pairs] = n;
	for(int i = 2*number_block_pairs; i-- > 0 ; )
	{
		device_index[i] = device_index[i+1] - (int)(device_index[i+1] / i) ;
	}
	
	// ��������� ������������ ������ ���������� ������ ���� �������� ������

	for (int i = 0; i < number_tasks; i++) {
		// ��������� ������������ �������� ����������� ����� � ������ ������ 
		// � ���������� ������������ ��������
		global_oddeven_worker <<< blocks, threads >>>(device_data, device_index, number_block_pairs, (i&1), size, direction);
	}

	// ���������� ���������� � �������� ������
	cudaMemcpy(data, device_data, n*size, cudaMemcpyDeviceToHost);

	// ����������� ������ �� ����������
	cudaFree(device_data);
	cudaFree(device_index);
}

// ������� �������
// ���������
//	����� ������� ������ 
//	����� ������� �������� ������ ������ 
//	���������� ��� ������
//  ׸������ ��������
//  ������ ������ ��������
//	����������� ����������
__global__ void global_oddeven_worker(
	bype * data, 
	int * index,
	int number_block_pairs,
	int parity,
	size_t size,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < number_block_pairs) {
		int n = indexes[2*number_block_pairs];
		
		// �������� � ��������� ����� ������ - left � right
		// ��� ����� ���� ��������������� ���� �� ������
		// ���� ���� ��������� � ������ � ����� �������

		int left = (2*id+parity) % (2*number_block_pairs);
		int right = (2*id+1+parity) % (2*number_block_pairs);
			
		int start = index[left];
		int len = ((left<right)?0:n)+(index[right+1]-index[left]);
					
		// ��������� ������� �������� ��� ���������� ����� �������
		(*_non_parallel_sort)(data, start, len, n, size, direction);
		
		// ����� ������ ���� ������ ����� �����
		if (left<right)	index[right] = index[left]+(int)(len/2);
	}
}

// ������������ ���� ������ � ������ ����������
__device__ void device_exchange(byte *x, byte *y, int count)
{
	for (int i = 0; i < count; i++) {
		byte ch = x[i]; x[i] = y[i]; y[i] = ch;
	}
}

// ����������� ������ ������� ������ � ������
__device__ void device_copy(byte *x, byte *y, int count)
{
	for (int i = 0; i < count; i++) {
		x[i] = y[i];
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

/////////////////////////////////////////////////////////////////
// ����������� ���������� ����� �������
// ����������� - ����������� ����������� ��������� � ������� ����� n
__device__ void device_bubble_sort(byte *arr, int index, int len, int n, size_t size, int direction)
{
	if (index+len <= n) {
		for(int i = index ; i < index+len-1 ; i++ ) {
			for(int j = i + 1 ; j < index+len ; j++ ) {
				int value = direction*(*_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
		}
	} else {
		for(int i = 0 ; i < ((index+len) % n) ; i++ ) {
			for(int j = i + 1 ; j <= ((index+len)%n) ; j++ ) {
				int value = direction*(*_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
			for(int j = index ; j < n ; j++ ) {
				int value = direction*(*_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
		}
		for(int i = index ; i < n-1 ; i++ ) {
			for(int j = i + 1 ; j < n ; j++ ) {
				int value = direction*(*_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
		}
	}
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
	_non_parallel_sort = device_bubble_sort;
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
