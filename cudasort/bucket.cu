/*
������� ���������� (bucket sort)
� ������� ��������� ��� ��������� ����������(Bucket sort) ����������� �������� ������������
����� �������� ������ ��������� ������(��������, ������) ���, ����� ��� �������� � ������ ���������
����� ���� ������ ������(��� ������), ��� � ����������.������ ���� ����� ����������� �������� ����
������������ ��������� ������� ������� ������, HPC, HTC, Cloud - computing, ��������� ������,
�������� ������� ���������� ��� �� ������� ���� ������.����� �������� �������� ������� � ������.��� ���� ����������
���������� �������� ����� ����������.
�������� ������� ������ � ������� ����������� ������, ��������� �� ����� ������� "��������" �
"�������� �������", ����������� ��� ���������� ��������, ���������� ���������, ������� ����������,
���������� �����, ���������� ��������.
� ������������ ������ ���������� ������ ������ ��������� ������������ ��������� ����.�����:
1) ���� ��������� �������� ������ �� ���������;
2) ���� ���������� �������� �� ������������� ��������;
3) ���� ������������ ������ ����� �������;
4) ����� ��������� ���� ����� ������;
5) ���� �������� ������ � ������.
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
// _non_parallel_sort - ������ ���������� ��� ������������� ����������� ����������.
// _parallel_sort - ������ ���������� � �������������� ����������� ����������

comparer _comparer;
indexator _indexator;
non_parallel_sort _non_parallel_sort;
parallel_sort _parallel_sort;

__host__ void host_bucket_sort(byte *data, int n, size_t size, int direction)
{
// data - ������ ������
// n - ���������� ��������� � �������� ������� ��� ����������
// size - ������ ������ �������� ������� � ������
// direction - ������ ���������� 
// -1 �������� ���������� �� ��������, 
//  1 �������� ���������� �� �����������
	
	cudaError_t err;
	byte *device_data;
	byte *device_bucket;
	int *device_sizes;

	// ��������� ����������e ���������� ������ � ��������� �����������

	int len = min(max(1,(int)log(log((double)n))),5);
	int index = 8*size-len;

	// ��������� ����������� ��������� �� ��������, ����

	int blocks = 1 << (len>>1);
	int threads = 1 << (len-(len>>1));

	assert((1<<len) == blocks*threads);

	// ��� ������ - �������� �������� ������ � ������ GPU 

	error = cudaMalloc((void**)&device_data, n*size);
	cudaMemcpy(device_data, data, n*size, cudaMemcpyHostToDevice);

	// ��� ������ - �������� ������ ��� ������� 

	error = cudaMalloc((void**)&device_bucket, (n<<len)*size);
	error = cudaMalloc((void**)&device_sizes, (1<<len)*sizeof(int));

	// ��� ��e��� - ��������� ��������

	global_bucket_worker_collect <<< blocks, threads >>>(device_data, device_bucket, device_sizes, n, index, len, size, direction);
	global_bucket_worker_sort <<< blocks, threads >>>(device_data, device_bucket, device_sizes, n, index, len, size, direction);
	global_bucket_worker_merge <<< 1, 1 >>>(device_data, device_bucket, device_sizes, n, index, len, size, direction);

	// ���������� ���������� � �������� ������

	cudaMemcpy(data, device_data, n*size, cudaMemcpyDeviceToHost);

	// ����������� ������ �� ����������

	cudaFree(device_data);
	cudaFree(device_bucket);
	cudaFree(device_sizes);
}

// ������� ��������
// ���������
//	����� ������� ������ 
//	����� ������� ������ 
//	����� ������� ������� ������ 
//	������ �������
//  ��������� �����������
//  ������ ������ ��������
//	����������� ����������
__global__ void global_bucket_worker_collect(
	bype * data, 
	byte * bucket,
	int * sizes,
	int n,
	int index,
	int len,
	size_t size,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < (1<<len)) {
		// �������� ������ � �������
		for(int i = 0; i < n; i++) {
			if (id==(*_indexator)(&data[i*size],index,len,size)) {
				device_copy(&data[i*size],&bucket[id*n+sizes[id]++)*size],size)
			}
		}
	}
}
__global__ void global_bucket_worker_sort(
	bype * data, 
	byte * bucket,
	int * sizes,
	int n,
	int index,
	int len,
	size_t size,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < (1<<len)) {
		(*_non_parallel_sort)(bucket, id*n, sizes[id], n, size, direction);
	}
}
__global__ void global_bucket_worker_merge(
	bype * data, 
	byte * bucket,
	int * sizes,
	int n,
	int index,
	int len,
	size_t size,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	assert(id == 0);
	
	byte * next = data;
	for(int i=0; i < (1 << len) ; i++ ) {
		device_copy(next,&bucket[i*n],sizes[i]*size);
		next = &next[sizes[i]*size];
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

// ����������� ������ ������� 
__device__ int device_integer_indexator(byte *x, int index, int len,  size_t size)
{
	// ����������� ������������� ����� �� len ��� � ������� index
	assert(size == sizeof(long))
	return min(m,(((*(long*)x)>>index)+(1<<(8*sizeof(long)-index)))^((1<<len)-1);
}

/////////////////////////////////////////////////////////////////
// ����������� ���������� ����� �������
// ����������� - ����������� ����������� ��������� � ������� ����� n
__device__ void device_bubble_sort(byte *arr, int index, int len, int n, size_t size, int direction)
{
	assert(len > 0);

	if (index+len <= n) {
		for(int i = index ; i < index+len-1 ; i++ ) {
			for(int j = i + 1 ; j < index+len ; j++ ) {
				int value = direction*(_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
		}
	} else {
		for(int i = 0 ; i < ((index+len) % n) ; i++ ) {
			for(int j = i + 1 ; j <= ((index+len)%n) ; j++ ) {
				int value = direction*(_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
			for(int j = index ; j < n ; j++ ) {
				int value = direction*(_comparer)(&arr[i*size],&arr[j*size],size);
				if (value < 0) device_exchange(&arr[i*size],&arr[j*size],size);
			}
		}
		for(int i = index ; i < n-1 ; i++ ) {
			for(int j = i + 1 ; j < n ; j++ ) {
				int value = direction*(_comparer)(&arr[i*size],&arr[j*size],size);
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
	_parallel_sort = host_bucket_sort;

	for(int n = 10 ; n < 10000 ; n *= 10 )
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
		for(int i = 0 ; ( i < (n-1) ) && check; i++ ) 
			check = (arr[i] <= arr[i+1]);

		std::cout << "n = " << n << "\t" << "time = " << (end - start) << "\t" << "results :" << (check ? "ok" : "fail") << std::endl;
	
		// ������������ ������
		free(arr);
	}

	cudaDeviceReset();

	exit(0);
}
