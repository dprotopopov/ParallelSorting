char *title = "bucket sort";
char *description = "Блочная сортировка (bucket sort)";
/*
В блочной карманной или корзинной сортировке(Bucket sort) сортируемые элементы распределены
между конечным числом отдельных блоков(карманов, корзин) так, чтобы все элементы в каждом следующем
блоке были всегда больше(или меньше), чем в предыдущем.Каждый блок затем сортируется отдельно либо
Параллельная обработка больших объёмов данных, HPC, HTC, Cloud - computing, хранилища данных,
файловые системы рекурсивно тем же методом либо другим.Затем элементы помещают обратно в массив.Для этой сортировки
характерно линейное время исполнения.
Алгоритм требует знаний о природе сортируемых данных, выходящих за рамки функций "сравнить" и
"поменять местами", достаточных для сортировки слиянием, сортировки пирамидой, быстрой сортировки,
сортировки Шелла, сортировки вставкой.
В параллельной версии сортировки каждую группу элементов обрабатывает отдельный узел.Далее:
1) узлы считывают входящие данные из хранилища;
2) хост группирует элементы по определенному признаку;
3) хост распределяет данные между клонами;
4) клоны сортируют свою часть данных;
5) хост собирает данные с клонов.
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
// Настроечные аттрибуты
// _comparer - функция сравнения двух элементов массива
// _indexator - функция определения номера корзины для элемента массива
// _non_parallel_sort - фунция сортировки без использования паралельных вычислений
// _parallel_sort - фунция сортировки с использованием паралельных вычислений

#define fn_comparer  device_comparer<long>
#define fn_indexator device_indexator<long>
#define fn_non_parallel_sort device_bubble_sort<long>
#define fn_parallel_sort host_bucket_sort<long>

template<class T>
__host__ void host_bucket_sort(T *data, int n, int direction)
{
// data - массив данных
// n - количество элементов в исходном массиве для сортировки
// direction - способ сортировки 
// -1 означает сортировку по убыванию, 
//  1 означает сортировку по возрастанию
	
	cudaError_t err;
	T *device_data;
	T *device_bucket;
	int *device_size;

	// Шаг первый - копируем исходный массив в память GPU 

	err = cudaMalloc((void**)&device_data, n*sizeof(T));
	cudaMemcpy(device_data, data, n*sizeof(T), cudaMemcpyHostToDevice);

	// Определим оптимальноe количество корзин и парамерты индексатора

	int number = 2; while (number<n && number<5) number++;
	
	int len = 1 ; while ((1<<len)<number) len++;
	int index = 8*sizeof(T)-len;

	// Шаг второй - выделяем память под корзины 
	
	err = cudaMalloc((void**)&device_bucket, (n<<len)*sizeof(T));
	err = cudaMalloc((void**)&device_size, (1<<len)*sizeof(int));

	// Определим оптимальное разбиения на процессы, нити

	int blocks = 1 << (len>>1);
	int threads = 1 << (len-(len>>1));

	// Шаг трeтий - применяем алгоритм

	assert((1<<len) == blocks*threads);

	global_bucket_worker_collect <<< blocks, threads >>>(device_data, device_bucket, device_size, n, index, len, direction);
	global_bucket_worker_sort <<< blocks, threads >>>(device_data, device_bucket, device_size, n, index, len, direction);
	global_bucket_worker_merge <<< 1, 1 >>>(device_data, device_bucket, device_size, n, index, len, direction);

	// Возвращаем результаты в исходный массив

	cudaMemcpy(data, device_data, n*sizeof(T), cudaMemcpyDeviceToHost);

	// Освобождаем память на устройстве

	cudaFree(device_data);
	cudaFree(device_bucket);
	cudaFree(device_size);

	err = err;
}

// Функция процесса
// Параметры
//	адрес массива данных 
//	адрес массива корзин 
//	адрес массива размера корзин 
//	Размер массива
//  Параметры индексатора
//  Размер одного элемента
//	Направление сортировки
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
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < (1<<len)) {
		sizes[id] = 0;
		// Набираем товары в корзину
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
	// Получаем идентификатор нити
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
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	assert(id == 0);
	id = id;

	T * next = data;
	for(int i=0; i < (1 << len) ; i++ ) {
		device_copy(next,&bucket[i*n],sizes[i]);
		next = &next[sizes[i]];
	}
}

// Перестановка двух блоков в памяти устройства
template<class T>
__device__ void device_exchange(T *x, T *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		T ch = x[i] ; x[i] = y[i] ; y[i] = ch;
	}
}

// Копирование одного участка памяти в другой
template<class T>
__device__ void device_copy(T *x, T *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		x[i] = y[i] ;
	}
}

// Функция сравнения данных xранимых в памяти как целых чисел типа long
template<class T>
__device__ int device_comparer(T *x, T *y)
{
	if ((*x)<(*y)) return 1;
	else if ((*x)>(*y)) return -1;
	else return 0;
}

// Определение номера карзины 
// Формируется положмтельное число из len бит с позиции index
template<class T>
__device__ int device_indexator(T *x, int index, int len)
{
	assert(index+len <= sizeof(T));
	return (int)((((*x) >> index) + (1 << (8 * sizeof(T)-index))) & ((1 << len) - 1));
}

/////////////////////////////////////////////////////////////////
// Пузырьковая сортировка части массива
// Особенность - поддерживат циклическую адресацию в массиве длины n
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
		// Создаём массив длины n чисел типа long
		long *arr = (long *)malloc(n*sizeof(long));

		float total_time = 0.0;
		bool check = true;

		for(int j = 0; j < tests ; j++ ) {
			// Заполняем массив псевдо-случайными значениями используя функцию rand
			for (int i = 0; i<n; i++) { arr[i] = rand(); }

			// Сортируем массив по возрастанию
		
			time_t start = time(NULL);
			fn_parallel_sort(arr, n, 1);
			time_t end = time(NULL);

			total_time += (end - start);

			// Проверяем
			for (int i = 0; (i < (n - 1)) && check; i++)
				check = (arr[i] <= arr[i + 1]);
		}
		std::cout << "array size = " << n << "\t" << "avg time = " << (total_time/tests) << "\t" << "check result = " << (check ? "ok" : "fail") << "\t";
		for (int i = 0; i<n && i<24; i++) std::cout << arr[i] << ","; std::cout << " ..." << std::endl;

		// Высвобождаем массив
		free(arr);
	}

	cudaDeviceReset();

	exit(0);
}