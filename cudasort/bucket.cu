char *title = "bucket sort";
char *description = "������� ���������� (bucket sort)";
/*
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

#define assert( bool ) 

template<class T> __device__ void device_exchange(T *x, T *y, int count);
template<class T>__device__ void device_copy(T *x, T *y, int count);
template<class T> __device__ int device_comparer(T *x, T *y);
template<class T> __device__ int device_indexator(T *x, int index, int len);
template<class T> __device__ void device_bubble_sort(T *data, int index, int len, int n, int direction);
template<class T> __global__ void global_bucket_worker_collect(T * data, T * bucket, int * sizes, int n, int index,	int len, int direction);
template<class T> __global__ void global_bucket_worker_sort(T * data, T * bucket, int * sizes, int n, int index,	int len, int direction);
template<class T> __global__ void global_bucket_worker_merge(T * data, T * bucket, int * sizes, int n, int index,	int len, int direction);
template<class T> __host__ void host_bucket_sort(T *data, int n, int direction);

////////////////////////////////////////////////////////////////////////////////////////////
// ����������� ���������
// _comparer - ������� ��������� ���� ��������� �������
// _indexator - ������� ����������� ������ ������� ��� �������� �������
// _non_parallel_sort - ������ ���������� ��� ������������� ����������� ����������
// _parallel_sort - ������ ���������� � �������������� ����������� ����������

#define fn_comparer  device_comparer<long>
#define fn_indexator device_indexator<long>
#define fn_non_parallel_sort device_bubble_sort<long>
#define fn_parallel_sort host_bucket_sort<long>

template<class T>
__host__ void host_bucket_sort(T *data, int n, int direction)
{
// data - ������ ������
// n - ���������� ��������� � �������� ������� ��� ����������
// direction - ������ ���������� 
// -1 �������� ���������� �� ��������, 
//  1 �������� ���������� �� �����������
	
	cudaError_t err;
	T *device_data;
	T *device_bucket;
	int *device_size;

	// ��� ������ - �������� �������� ������ � ������ GPU 

	err = cudaMalloc((void**)&device_data, n*sizeof(T));
	cudaMemcpy(device_data, data, n*sizeof(T), cudaMemcpyHostToDevice);

	// ��������� ����������e ���������� ������ � ��������� �����������

	int number = 2; while (number<n && number<5) number++;
	
	int len = 1 ; while ((1<<len)<number) len++;
	int index = 8*sizeof(T)-len;

	// ��� ������ - �������� ������ ��� ������� 
	
	err = cudaMalloc((void**)&device_bucket, (n<<len)*sizeof(T));
	err = cudaMalloc((void**)&device_size, (1<<len)*sizeof(int));

	// ��������� ����������� ��������� �� ��������, ����

	int blocks = 1 << (len>>1);
	int threads = 1 << (len-(len>>1));

	// ��� ��e��� - ��������� ��������

	assert((1<<len) == blocks*threads);

	global_bucket_worker_collect <<< blocks, threads >>>(device_data, device_bucket, device_size, n, index, len, direction);
	global_bucket_worker_sort <<< blocks, threads >>>(device_data, device_bucket, device_size, n, index, len, direction);
	global_bucket_worker_merge <<< 1, 1 >>>(device_data, device_bucket, device_size, n, index, len, direction);

	// ���������� ���������� � �������� ������

	cudaMemcpy(data, device_data, n*sizeof(T), cudaMemcpyDeviceToHost);

	// ����������� ������ �� ����������

	cudaFree(device_data);
	cudaFree(device_bucket);
	cudaFree(device_size);

	err = err;
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
template<class T>
__global__ void global_bucket_worker_collect(
	T * data, 
	T * bucket,
	int * sizes,
	int n,
	int index,
	int len,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < (1<<len)) {
		sizes[id] = 0;
		// �������� ������ � �������
		for(int i = 0; i < n; i++) {
			if (id==fn_indexator(&data[i],index,len)) {
				device_copy(&bucket[id*n+sizes[id]++],&data[i],1);
			}
		}
	}
}
template<class T>
__global__ void global_bucket_worker_sort(
	T * data, 
	T * bucket,
	int * sizes,
	int n,
	int index,
	int len,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < (1<<len)) {
		fn_non_parallel_sort(bucket, id*n, sizes[id], n, direction);
	}
}
template<class T>
__global__ void global_bucket_worker_merge(
	T * data, 
	T * bucket,
	int * sizes,
	int n,
	int index,
	int len,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	assert(id == 0);
	id = id;

	T * next = data;
	for(int i=0; i < (1 << len) ; i++ ) {
		device_copy(next,&bucket[i*n],sizes[i]);
		next = &next[sizes[i]];
	}
}

// ������������ ���� ������ � ������ ����������
template<class T>
__device__ void device_exchange(T *x, T *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		T ch = x[i] ; x[i] = y[i] ; y[i] = ch;
	}
}

// ����������� ������ ������� ������ � ������
template<class T>
__device__ void device_copy(T *x, T *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		x[i] = y[i] ;
	}
}

// ������� ��������� ������ x������� � ������ ��� ����� ����� ���� long
template<class T>
__device__ int device_comparer(T *x, T *y)
{
	if ((*x)<(*y)) return 1;
	else if ((*x)>(*y)) return -1;
	else return 0;
}

// ����������� ������ ������� 
// ����������� ������������� ����� �� len ��� � ������� index
template<class T>
__device__ int device_indexator(T *x, int index, int len)
{
	assert(index+len <= sizeof(T));
	return (int)((((*x) >> index) + (1 << (8 * sizeof(T)-index))) & ((1 << len) - 1));
}

/////////////////////////////////////////////////////////////////
// ����������� ���������� ����� �������
// ����������� - ����������� ����������� ��������� � ������� ����� n
template<class T>
__device__ void device_bubble_sort(T *data, int index, int len, int n, int direction)
{
	if (index+len <= n) {
		for(int i = index ; i < index+len-1 ; i++ ) {
			for(int j = i + 1 ; j < index+len ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value < 0) device_exchange<T>(&data[i],&data[j],1);
			}
		}
	} else {
		for(int i = 0 ; i < ((index+len) % n) ; i++ ) {
			for(int j = i + 1 ; j <= ((index+len)%n) ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value < 0) device_exchange<T>(&data[i],&data[j],1);
			}
			for(int j = index ; j < n ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value < 0) device_exchange<T>(&data[i],&data[j],1);
			}
		}
		for(int i = index ; i < n-1 ; i++ ) {
			for(int j = i + 1 ; j < n ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value < 0) device_exchange<T>(&data[i],&data[j],1);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	std::cout << title << std::endl;

	// Find/set the device.
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
	}

	for (int n = 100, tests = 100; n <= 1000; n += 100, tests = ((tests>>1)+10))
	{
		// ������ ������ ����� n ����� ���� long
		long *arr = (long *)malloc(n*sizeof(long));

		float total_time = 0.0;
		bool check = true;

		for(int j = 0; j < tests ; j++ ) {
			// ��������� ������ ������-���������� ���������� ��������� ������� rand
			for (int i = 0; i<n; i++) { arr[i] = rand(); }

			// ��������� ������ �� �����������
		
			time_t start = time(NULL);
			fn_parallel_sort(arr, n, 1);
			time_t end = time(NULL);

			total_time += (end - start);

			// ���������
			for (int i = 0; (i < (n - 1)) && check; i++)
				check = (arr[i] <= arr[i + 1]);
		}
		std::cout << "array size = " << n << "\t" << "avg time = " << (total_time/tests) << "\t" << "check result = " << (check ? "ok" : "fail") << "\t";
		for (int i = 0; i<n && i<24; i++) std::cout << arr[i] << ","; std::cout << " ..." << std::endl;

		// ������������ ������
		free(arr);
	}

	cudaDeviceReset();

	exit(0);
}