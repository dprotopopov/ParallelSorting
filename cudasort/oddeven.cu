/*
Сортировка четно-нечетными перестановками (odd-even sort)
Для каждой итерации алгоритма операции сравнения-обмена для всех пар элементов независимы и
выполняются одновременно. Рассмотрим случай, когда число процессоров равно числу элементов, т.е. p=n -
число процессоров (сортируемых элементов). Предположим, что вычислительная система имеет топологию
кольца. Пусть элементы ai (i = 1, .. , n), первоначально расположены на процессорах pi (i = 1, ... , n). В нечетной
итерации каждый процессор с нечетным номером производит сравнение-обмен своего элемента с элементом,
находящимся на процессоре-соседе справа. Аналогично в течение четной итерации каждый процессор с четным
номером производит сравнение-обмен своего элемента с элементом правого соседа.
На каждой итерации алгоритма нечетные и четные процессоры выполняют шаг сравнения-обмена с их
правыми соседями за время Q(1). Общее количество таких итераций – n; поэтому время выполнения
параллельной сортировки – Q(n).
Когда число процессоров p меньше числа элементов n, то каждый из процессов получает свой блок
данных n/p и сортирует его за время Q((n/p)·log(n/p)). Затем процессоры проходят p итераций (р/2 и чётных, и
нечётных) и делают сравнивания-разбиения: смежные процессоры передают друг другу свои данные, а
внутренне их сортируют (на каждой паре процессоров получаем одинаковые массивы). Затем удвоенный
массив делится на 2 части; левый процессор обрабатывает далее только левую часть (с меньшими значениями
данных), а правый – только правую (с большими значениями данных). Получаем отсортированный массив
после p итераций, выполняя в каждой такие шаги:
23)узлы считывают входные данные из хранилища;
24)хост распределяет данные между клонами;
25)клоны сортируют свою часть данных;
26)клоны обмениваются частями, при этом объединяя четные и нечетные части;
27)если количество перестановок равно количеству клонов, то алгоритм завершен;
28)хост формирует отсортированный массив.

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
// _non_parallel_sort - фунция сортировки без использования паралельных вычислений
// _parallel_sort - фунция сортировки с использованием паралельных вычислений

comparer _comparer;
indexator _indexator;
non_parallel_sort _non_parallel_sort;
parallel_sort _parallel_sort;

__host__ void host_oddeven_sort(byte *data, int n, size_t size, int direction)
{
// data - массив данных
// n - количество элементов в исходном массиве для сортировки
// size - размер одного элемента массива в байтах
// direction - способ сортировки 
// -1 означает сортировку по убыванию, 
//  1 означает сортировку по возрастанию
	
	cudaError_t err;
	byte *device_data;
	int * device_index;

	// Определение оптимального разбиения на блоки, процессы и нити
	// одна нить в просессе будет сортировать массив длины 2*block_length

	int block_length = (int)((max(1,(int)pow((double)n,0.33333333333))+1)/2);

	// Для реализации алгоритма нам потребуется массив состоящий из пар блоков, 
	// то есть количество блоков должно быть кратно 2 
	// поскольку на одном шаге сортировки все блоки разбиваются на пары 
	
	int number_block_pairs = (int)((n+2*block_length-1)/(2*block_length));

	// Определим оптимальное разбиения на процессы, нити
	// одна нить в просессе будет сортировать массив длины 2*block_length

	int blocks = max(1,(int)sqrt((double)number_block_pairs))),255);
	int threads = (int)((number_block_pairs+number_tasks-1)/number_tasks);

	assert(n <= 2*block_length*number_block_pairs);
	assert(number_block_pairs <= number_tasks*number_threads);
	
	// Шаг первый - копируем исходный массив в память GPU 

	error = cudaMalloc((void**)&device_data, n*size);
	cudaMemcpy(device_data, data, n*size, cudaMemcpyHostToDevice);

	// Шаг второй - разделим все элементы массива между блоками
	// соответственно они будут содержать данные разной длины, 
	// и нам понадобится 1 вспомогательный массив
	// с количеством элементов в предыдущих блоках

	error = cudaMalloc((void**)&device_index, (2*number_block_pairs+1)*sizeof(int));

	// Равномерно распределим исходные данные между блоками
	device_index[2*number_block_pairs] = n;
	for(int i = 2*number_block_pairs; i-- > 0 ; )
	{
		device_index[i] = device_index[i+1] - (int)(device_index[i+1] / i) ;
	}
	
	// запускаем параллельные задачи сортировки данных двух соседних блоков

	for (int i = 0; i < number_tasks; i++) {
		// Запускаем параллельные процессы копирования левых и правых блоков 
		// и сортировки получившихся массивов
		global_oddeven_worker <<< blocks, threads >>>(device_data, device_index, number_block_pairs, (i&1), size, direction);
	}

	// Возвращаем результаты в исходный массив
	cudaMemcpy(data, device_data, n*size, cudaMemcpyDeviceToHost);

	// Освобождаем память на устройстве
	cudaFree(device_data);
	cudaFree(device_index);
}

// Рабочий процесс
// Параметры
//	адрес массива данных 
//	адрес массива индексов блоков данных 
//	Количество пар блоков
//  Чётность операции
//  Размер одного элемента
//	Направление сортировки
__global__ void global_oddeven_worker(
	bype * data, 
	int * index,
	int number_block_pairs,
	int parity,
	size_t size,
	int direction)
{
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < number_block_pairs) {
		int n = indexes[2*number_block_pairs];
		
		// Работаем с следующей парой блоков - left и right
		// Они могут идти последовательно друг за другом
		// либо быть разнесены в начало и конец массива

		int left = (2*id+parity) % (2*number_block_pairs);
		int right = (2*id+1+parity) % (2*number_block_pairs);
			
		int start = index[left];
		int len = ((left<right)?0:n)+(index[right+1]-index[left]);
					
		// Запускаем обычный алгоритм для сортировки части массива
		(*_non_parallel_sort)(data, start, len, n, size, direction);
		
		// Делим данные двух блоков между собой
		if (left<right)	index[right] = index[left]+(int)(len/2);
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

/////////////////////////////////////////////////////////////////
// Пузырьковая сортировка части массива
// Особенность - поддерживат циклическую адресацию в массиве длины n
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
		for (int i = 0; (i < (n - 1)) && check; i++)
			check = (arr[i] <= arr[i + 1]);

		std::cout << "n = " << n << "\t" << "time = " << (end - start) << "\t" << "results :" << (check ? "ok" : "fail") << std::endl;

		// Высвобождаем массив
		free(arr);
	}

	cudaDeviceReset();

	exit(0);
}
