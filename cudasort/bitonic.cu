char *title = "bitonic sort";
char *description = "������������ ���������� (bitonic sort)";
/*
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

typedef int(*comparer)(void *x, void *y);
typedef int(*indexator)(void *x, int index, int len);
typedef void(*non_parallel_sort)(void *arr, int index, int len, int n, int direction);
typedef void(*parallel_sort)(void *data, int n, int direction);

#define assert( bool ) 

template<class T> __device__ void device_exchange(T *x, T *y, int count);
template<class T>__device__ void device_copy(T *x, T *y, int count);
template<class T> __device__ int device_comparer(T *x, T *y);
template<class T> __device__ int device_indexator(T *x, int index, int len);
template<class T> __device__ void device_bubble_sort(T *data, int index, int len, int n, int direction);
template<class T> __device__ void device_bitonic_sort(T *data, int index, int len, int n, int direction);
template<class T> __global__ void global_bitonic_worker(T * data, int n, int i, int j, int loops, int direction);
template<class T> __global__ void global_bitonic_merger(T * data, T * data2, int * sizes, int n, int direction);
template<class T> __host__ void host_bitonic_sort(T *data, int n, int direction);

////////////////////////////////////////////////////////////////////////////////////////////
// ����������� ���������
// _comparer - ������� ��������� ���� ��������� �������
// _indexator - ������� ����������� ������ ������� ��� �������� �������
// _non_parallel_sort - ������ ���������� ��� ������������� ����������� ����������
// _parallel_sort - ������ ���������� � �������������� ����������� ����������

#define fn_comparer  device_comparer<long>
#define fn_indexator device_indexator<long>
#define fn_non_parallel_sort device_bubble_sort<long>
#define fn_parallel_sort host_bitonic_sort<long>

template<class T>
__host__ void host_bitonic_sort(T *data, int n, int direction)
{
	// data - ������ ������
	// n - ���������� ��������� � �������� ������� ��� ����������
	// direction - ������ ���������� 
	// -1 �������� ���������� �� ��������, 
	//  1 �������� ���������� �� �����������

	cudaError_t err;
	T *device_data;
	T *device_data2;
	int *device_size;

	// ����� ���� ��������� k*(k-1)/2*2^(k-1) �������� ���������, ��� k = log2 n
	// �� ���� �������� ������� ��������� ����� ��������� n/2 = 2^(k-1) ��������

	// ��� ������ - �������� �������� ������ � ������ GPU 

	err = cudaMalloc((void**)&device_data, n*sizeof(T));
	cudaMemcpy(device_data, data, n*sizeof(T), cudaMemcpyHostToDevice);

	// ����� n ����������� � ���� ����� �������� ������,
	// �������, ��������� �������� ������ �� ���������� � ������� ������� ��������� ���� �����
	// � ��������� ������ ��������� ������������ ���������� 
	// � ���������� ������� ������ ����� ���������� ��������������� �������� ������� ������ �������� ������

	for(int k=1; (1<<k) <= n ; k++) {

		if ( n & (1<<k) ) {

			for(int i = 0; i < k ; i++ ) {
				for( int j = i; j >= 0 ; j-- ) 	{ 

					// ��������� ����������� ��������� �� ��������, ���� � ����� 
					// ���� ���� � �������� ����� ����� ��������� ����� � ��������� ����������� ��������

					int blocks = 1 << (max(1,(int)k/3));
					int threads = 1 << (max(1,(int)k/3));
					int loops = 1 << (k-2*max(1,(int)k/3)-1);

					assert((1<<k) == 2*blocks*threads*loops);

					// ���������� ��� � ������ ����� ����������� ���������� �������� (�������������� ������� � ����� � ��� �� ������)
					global_bitonic_worker<T> <<< blocks, threads >>>(&device_data[n&((1<<k)-1)], n&(1<<k), i, j, loops, direction);
				}
			}
		}
	}

	// ������ ���� ���������� ������� ��� ��������������� ��������
	// ��� ����� �������� ������ ������ �� ������� ��� � ������
	// � ������ �������� ��������

	err = cudaMalloc((void**)&device_data2, n*sizeof(T));
	err = cudaMalloc((void**)&device_size, sizeof(int)*sizeof(int)*8);

	global_bitonic_merger<T> <<< 1, 1 >>>(device_data, device_data2, device_size, n , direction);

	// ���������� ���������� � �������� ������
	cudaMemcpy(data, device_data2, n*sizeof(T), cudaMemcpyDeviceToHost);

	// ����������� ������ �� ����������
	cudaFree((void*)device_data);
	cudaFree((void*)device_data2);
	cudaFree((void*)device_size);
	
	err = err;
}

template<class T>
__global__ void global_bitonic_merger(
	T * data, 
	T * data2, 
	int * size,
	int n,
	int direction)
{
	for(int k=0; k<8*sizeof(int) ; k++ ) size[k] = n & (1<<k);

	int total = n;

	while(total > 0) {
		int k = 8*sizeof(int);	while( (k-->0) && (size[k] == 0) ) ;
		for (int i=k; i-- ; ) {
			if (size[i] > 0 &&	
				direction*fn_comparer(
					&data[(n&((1<<k)-1))+size[k]-1],
					&data[(n&((1<<i)-1))+size[i]-1]) > 0)
			{
				k = i;
			}
		}
		total--;
		size[k]--;
		device_copy(&data2[total],&data[(n&((1<<k)-1))+size[k]],1);
	}
}

template<class T>
__global__ void global_bitonic_worker(
	T * data, 
	int n, int i, int j,
	int loops,
	int direction)
{
	// �������� ������������� ����
	int block = blockDim.x*blockIdx.x + threadIdx.x;
	int step = 1<<j;
	for(int y=0; y<loops; y++) {
		// �������� ������������� ���� �����
		int id = block*loops+y;
		int offset = ((id>>j)<<(j+1))+(id&((1<<j)-1));
		int parity = id >> i;
		while(parity>1) parity = (parity>>1) ^ (parity&1);
		parity = 1-(parity<<1); // ������ ���������� parity ����� ����� ������ 2 �������� 1 � -1

		assert ((offset+step) < n) ;
		
		int value = parity*direction*fn_comparer(&data[offset],&data[offset+step]);
		if (value < 0) device_exchange<T>(&data[offset],&data[offset+step],1);
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

	for (int n = 10000, tests = 10; n <= 100000; n += 10000, tests = ((tests>>1)+1))
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