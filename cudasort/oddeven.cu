char *title = "odd-even sort";
char *description = "���������� �����-��������� �������������� (odd-even sort)";
/*
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

#define assert( bool ) 

template<class T> __device__ void device_exchange(T *x, T *y, int count);
template<class T>__device__ void device_copy(T *x, T *y, int count);
template<class T> __device__ int device_comparer(T *x, T *y);
template<class T> __device__ int device_indexator(T *x, int index, int len);
template<class T> __device__ void device_bubble_sort(T *data, int index, int len, int n, int direction);
template<class T> __global__ void global_oddeven_spliter(int * index,	int n, int block_pairs);
template<class T> __global__ void global_oddeven_worker(T * data, int * index, int block_pairs, int parity, int direction);
template<class T> __host__ void host_oddeven_sort(T *data, int n, int direction);

////////////////////////////////////////////////////////////////////////////////////////////
// ����������� ���������
// _comparer - ������� ��������� ���� ��������� �������
// _indexator - ������� ����������� ������ ������� ��� �������� �������
// _non_parallel_sort - ������ ���������� ��� ������������� ����������� ����������
// _parallel_sort - ������ ���������� � �������������� ����������� ����������

#define fn_comparer  device_comparer<long>
#define fn_indexator device_indexator<long>
#define fn_non_parallel_sort device_bubble_sort<long>
#define fn_parallel_sort host_oddeven_sort<long>

template<class T>
__host__ void host_oddeven_sort(T *data, int n, int direction)
{
// data - ������ ������
// n - ���������� ��������� � �������� ������� ��� ����������
// direction - ������ ���������� 
// -1 �������� ���������� �� ��������, 
//  1 �������� ���������� �� �����������
	
	cudaError_t err;
	T *device_data;
	int * device_index;

	// ����������� ������������ ��������� �� ����������

	int block_length = max(1,(int)pow((double)n,0.33333333333));

	// ��� ���������� ��������� ��� ����������� ������ ��������� �� ��� ������, 
	// �� ���� ���������� ������ ������ ���� ������ 2 
	// ��������� �� ����� ���� ���������� ��� ����� ����������� �� ���� 
	
	int block_pairs = (int)((n+(2*block_length)-1)/(2*block_length));

	// ��������� ����������� ��������� �� ��������, ����
	// ���� ���� � �������� ����� ����������� ������ ����� 2*block_length

	int blocks = min(max(1,(int)sqrt((double)block_pairs)),255);
	int threads = (int)((block_pairs+blocks-1)/blocks);

	assert(n <= 2*block_length*block_pairs);
	assert(block_pairs <= blocks*threads);
	
	// ��� ������ - �������� �������� ������ � ������ GPU 

	err = cudaMalloc((void**)&device_data, n*sizeof(T));
	cudaMemcpy(device_data, data, n*sizeof(T), cudaMemcpyHostToDevice);

	// ��� ������ - �������� ��� �������� ������� ����� �������
	// �������������� ��� ����� ��������� ������ ������ �����, 
	// � ��� ����������� 1 ��������������� ������
	// � ����������� ��������� � ���������� ������

	err = cudaMalloc((void**)&device_index, (2*block_pairs+1)*sizeof(int));

	// ���������� ����������� �������� ������ ����� �������

	global_oddeven_spliter<T> <<< 1, 1 >>>( device_index, n, block_pairs );
	
	// ��������� ������������ ������ ���������� ������ ���� �������� ������

	for (int i = 0; i < 2*block_pairs; i++ ) {
		// ��������� ������������ �������� ����������� ����� � ������ ������ 
		// � ���������� ������������ ��������
		global_oddeven_worker<T> <<< blocks, threads >>>(device_data, device_index, block_pairs, (i&1), direction);
	}

	// ���������� ���������� � �������� ������
	cudaMemcpy(data, device_data, n*sizeof(T), cudaMemcpyDeviceToHost);

	// ����������� ������ �� ����������
	cudaFree(device_data);
	cudaFree(device_index);

	err = err;
}

template<class T>
__global__ void global_oddeven_spliter(
	int * index,
	int n,
	int block_pairs)
{
	index[2*block_pairs] = n;
	for(int i = 2*block_pairs; i > 0 ; i-- )
	{
		index[i-1] = index[i] - (int)(index[i] / i) ;
	}
}

// ������� �������
// ���������
//	����� ������� ������ 
//	����� ������� �������� ������ ������ 
//	���������� ��� ������
//  ׸������ ��������
//  ������ ������ ��������
//	����������� ����������
template<class T>
__global__ void global_oddeven_worker(
	T * data, 
	int * index,
	int block_pairs,
	int parity,
	int direction)
{
	// �������� ������������� ����
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < block_pairs) {
		int n = index[2*block_pairs];
		
		// �������� � ��������� ����� ������ - left � right
		// ��� ����� ���� ��������������� ���� �� ������
		// ���� ���� ��������� � ������ � ����� �������

		int left = (2*id+parity) % (2*block_pairs);
		int right = (2*id+1+parity) % (2*block_pairs);
			
		int start = index[left];
		int len = ((left<right)?0:n)+(index[right+1]-index[left]);
					
		// ��������� ������� �������� ��� ���������� ����� �������
		fn_non_parallel_sort(data, start, len, n, direction);
		
		// ����� ������ ���� ������ ����� �����
		if (left<right)	index[right] = index[left]+(int)(len/2);
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

	for (int n = 1000, tests = 100; n <= 10000; n += 1000, tests = ((tests>>1)+1))
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