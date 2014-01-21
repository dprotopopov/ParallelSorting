/*
Блочная сортировка (bucket sort)
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

typedef byte char;
typedef size_t int;
typedef __device__ int(*comparer)(byte *x, byte *y, size_t size);
typedef __device__ int(*indexator)(byte *x, int index, int len, size_t size);
typedef __device__ void(*non_parallel_sort)(byte *arr, int index, int len, int n, size_t size, int direction);
typedef __host__ void(*parallel_sort)(byte *data, int n, size_t size, int direction);

////////////////////////////////////////////////////////////////////////////////////////////
// Настроечные аттрибуты
// _comparer - функция сравнения двух элементов массива
// _indexator - функция определения номера корзины для элемента массива
// _non_parallel_sort - фунция сортировки без использования паралельных вычислений.
// _parallel_sort - фунция сортировки с использованием паралельных вычислений

comparer _comparer;
indexator _indexator;
non_parallel_sort _non_parallel_sort;
parallel_sort _parallel_sort;

__host__ void host_bucket_sort(byte *data, int n, size_t size, int direction)
{
// data - массив данных
// n - количество элементов в исходном массиве для сортировки
// size - размер одного элемента массива в байтах
// direction - способ сортировки 
// -1 означает сортировку по убыванию, 
//  1 означает сортировку по возрастанию
	
	cudaError_t err;
	byte *device_data;
	byte *device_bucket;
	int *device_sizes;

	// Определим оптимальноe количество корзин и парамерты индексатора

	int len = min(max(1,(int)log(log((double)n))),5);
	int index = 8*size-len;

	// Определим оптимальное разбиения на процессы, нити

	int blocks = 1 << (len>>1);
	int threads = 1 << (len-(len>>1));

	assert((1<<len) == blocks*threads);

	// Шаг первый - копируем исходный массив в память GPU 

	error = cudaMalloc((void**)&device_data, n*size);
	cudaMemcpy(device_data, data, n*size, cudaMemcpyHostToDevice);

	// Шаг второй - выделяем память под корзины 

	error = cudaMalloc((void**)&device_bucket, (n<<len)*size);
	error = cudaMalloc((void**)&device_sizes, (1<<len)*sizeof(int));

	// Шаг трeтий - применяем алгоритм

	global_bucket_worker_collect <<< blocks, threads >>>(device_data, device_bucket, device_sizes, n, index, len, size, direction);
	global_bucket_worker_sort <<< blocks, threads >>>(device_data, device_bucket, device_sizes, n, index, len, size, direction);
	global_bucket_worker_merge <<< 1, 1 >>>(device_data, device_bucket, device_sizes, n, index, len, size, direction);

	// Возвращаем результаты в исходный массив

	cudaMemcpy(data, device_data, n*size, cudaMemcpyDeviceToHost);

	// Освобождаем память на устройстве

	cudaFree(device_data);
	cudaFree(device_bucket);
	cudaFree(device_sizes);
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
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < (1<<len)) {
		// Набираем товары в корзину
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
	// Получаем идентификатор нити
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
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	assert(id == 0);
	
	byte * next = data;
	for(int i=0; i < (1 << len) ; i++ ) {
		device_copy(next,&bucket[i*n],sizes[i]*size);
		next = &next[sizes[i]*size];
	}
}

// Перестановка двух блоков в памяти устройства
__device__ void device_exchange(byte *x, byte *y, int count)
{
	for (int i = 0; i < count; i++) {
		byte ch = x[i]; x[i] = y[i]; y[i] = ch;
	}
}

// Копирование одного участка памяти в другой
__device__ void device_copy(byte *x, byte *y, int count)
{
	for (int i = 0; i < count; i++) {
		x[i] = y[i];
	}
}

// Функция сравнения данных xранимых в памяти как целых чисел типа long
__device__ int device_integer_comparer(byte *x, byte *y, size_t size)
{
	assert(size == sizeof(long));
	if ((*(long*)x)<(*(long*)y)) return 1;
	else if ((*(long*)x)>(*(long*)y)) return -1;
	else return 0;
}

// Определение номера карзины 
__device__ int device_integer_indexator(byte *x, int index, int len,  size_t size)
{
	// Формируется положмтельное число из len бит с позиции index
	assert(size == sizeof(long))
	return min(m,(((*(long*)x)>>index)+(1<<(8*sizeof(long)-index)))^((1<<len)-1);
}

/////////////////////////////////////////////////////////////////
// Пузырьковая сортировка части массива
// Особенность - поддерживат циклическую адресацию в массиве длины n
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
		// Создаём массив длины n чисел типа long
		// Заполняем массив псевдо-случайными значениями используя функцию rand

		long *arr = (long *)malloc(n*sizeof(long));
		for (int i = 0; i<n; i++) { arr[i] = rand(); }

		// Сортируем массив по возрастанию
		time_t start = time(NULL);
		(*_parallel_sort)(arr, n, sizeof(long), 1);
		time_t end = time(NULL);

		// Проверяем
		bool check = true;
		for(int i = 0 ; ( i < (n-1) ) && check; i++ ) 
			check = (arr[i] <= arr[i+1]);

		std::cout << "n = " << n << "\t" << "time = " << (end - start) << "\t" << "results :" << (check ? "ok" : "fail") << std::endl;
	
		// Высвобождаем массив
		free(arr);
	}

	cudaDeviceReset();

	exit(0);
}
