char *title = "odd-even sort";
char *description = "Сортировка четно-нечетными перестановками (odd-even sort)";
/*
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
после p итераций
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
template<class T> __global__ void global_oddeven_preworker(T * data, int * index, int block_pairs, int direction);
template<class T> __global__ void global_oddeven_worker(T * data, int * index, int block_pairs, int parity, int direction);
template<class T> __host__ void host_oddeven_sort(T *data, int n, int direction);

////////////////////////////////////////////////////////////////////////////////////////////
// Настроечные аттрибуты
// _comparer - функция сравнения двух элементов массива
// _indexator - функция определения номера корзины для элемента массива
// _non_parallel_sort - фунция сортировки без использования паралельных вычислений
// _parallel_sort - фунция сортировки с использованием паралельных вычислений

#define fn_comparer  device_comparer<long>
#define fn_indexator device_indexator<long>
#define fn_non_parallel_sort device_bubble_sort<long>
#define fn_parallel_sort host_oddeven_sort<long>

template<class T>
__host__ void host_oddeven_sort(T *data, int n, int direction)
{
// data - массив данных
// n - количество элементов в исходном массиве для сортировки
// direction - способ сортировки 
// -1 означает сортировку по убыванию, 
//  1 означает сортировку по возрастанию
	
	cudaError_t err;
	T *device_data[2];
	int * device_index;

	// Определение оптимального разбиения на подмассивы

	int block_length = max(1,(int)pow((double)n,0.33333333333));

	// Для реализации алгоритма нам потребуется массив состоящий из пар блоков, 
	// то есть количество блоков должно быть кратно 2 
	// поскольку на одном шаге сортировки все блоки разбиваются на пары 
	
	int block_pairs = (int)((n+(2*block_length)-1)/(2*block_length));

	// Определим оптимальное разбиения на процессы, нити
	// одна нить в просессе будет сортировать массив длины 2*block_length

	int blocks = min(max(1,(int)sqrt((double)block_pairs)),255);
	int threads = (int)((block_pairs+blocks-1)/blocks);

	assert(n <= 2*block_length*block_pairs);
	assert(block_pairs <= blocks*threads);
	
	// Шаг первый - копируем исходный массив в память GPU 

	err = cudaMalloc((void**)&device_data[0], n*sizeof(T));
	err = cudaMalloc((void**)&device_data[1], n*sizeof(T));

	cudaMemcpy(device_data[0], data, n*sizeof(T), cudaMemcpyHostToDevice);

	// Шаг второй - разделим все элементы массива между блоками
	// соответственно они будут содержать данные разной длины, 
	// и нам понадобится 1 вспомогательный массив
	// с количеством элементов в предыдущих блоках

	err = cudaMalloc((void**)&device_index, (2*block_pairs+1)*sizeof(int));

	// Равномерно распределим исходные данные между блоками

	global_oddeven_spliter<T> <<< 1, 1 >>>( device_index, n, block_pairs );
	
	// запускаем параллельные задачи сортировки данных двух соседних блоков

	global_oddeven_preworker<T> <<< blocks, 2*threads >>>(device_data[0], device_index, block_pairs, direction);

	for (int i = 0; i < 2*block_pairs; i++ ) {
		// Запускаем параллельные процессы копирования левых и правых блоков 
		// и сортировки получившихся массивов
		global_oddeven_worker<T> <<< blocks, threads >>>(device_data[i&1],device_data[1-(i&1)], device_index, block_pairs, (i&1), direction);
	}

	// Возвращаем результаты в исходный массив
	cudaMemcpy(data, device_data[1], n*sizeof(T), cudaMemcpyDeviceToHost);

	// Освобождаем память на устройстве
	cudaFree(device_data[1]);
	cudaFree(device_data[0]);
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

template<class T>
__global__ void global_oddeven_preworker(
	T * data, 
	int * index,
	int block_pairs,
	int direction)
{
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < 2*block_pairs) {
		int n = index[2*block_pairs];
		
		// Сортируем массив id

		int start = index[id];
		int len = index[id+1]-index[id];
					
		// Запускаем обычный алгоритм для сортировки части массива
		fn_non_parallel_sort(data, start, len, n, direction);
	}
}
// Рабочий процесс
// Параметры
//	адрес исходного массива данных 
//	адрес результирующего массива данных 
//	адрес массива индексов блоков данных 
//	Количество пар блоков
//  Чётность операции
//  Размер одного элемента
//	Направление сортировки
template<class T>
__global__ void global_oddeven_worker(
	T * data0, 
	T * data1, 
	int * index,
	int block_pairs,
	int parity,
	int direction)
{
	// Получаем идентификатор нити
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	if (id < block_pairs) {
		// Работаем с следующей парой блоков - left и right
		// Они могут идти последовательно друг за другом
		// либо быть разнесены в начало и конец массива

		int left = (2*id+parity) % (2*block_pairs);
		int right = (2*id+1+parity) % (2*block_pairs);
			
		int index_left = index[left];
		int index_right = index[right];
		int size_left = index[left+1]-index[left];
		int size_right = index[right+1]-index[right];
		int start0 = (left<right)?index_left:index_right;
		int start1 = (left>right)?index_left:index_right;
		int total0 = (left<right)?size_left:size_right;
		int total1 = (left>right)?size_left:size_right;

		// Запускаем алгоритм для слияния отсортированных частей массива
		while(size_left > 0 && size_right > 0) {
			int value = direction*fn_comparer(&data0[index_left+size_left-1],&data0[index_right+size_right-1]);
			if (value < 0) {
				device_copy<T>(&data1[(total1>0)?(start1+total1-1):(start0+total0-1)],&data0[index_right+(--size_right)],1);
			} else {
				device_copy<T>(&data1[(total1>0)?(start1+total1-1):(start0+total0-1)],&data0[index_left+(--size_left)],1);
			} 
			if (total1 > 0) total1--; else total0--;
		}
		while(size_left > 0) {
			device_copy<T>(&data1[(total1>0)?(start1+total1-1):(start0+total0-1)],&data0[index_left+(--size_left)],1);
			if (total1 > 0) total1--; else total0--;
		}
		while(size_right > 0) {
			device_copy<T>(&data1[(total1>0)?(start1+total1-1):(start0+total0-1)],&data0[index_right+(--size_right)],1);
			if (total1 > 0) total1--; else total0--;
		}
		
		// Делим данные двух блоков между собой
		if (left<right)	index[right] = (index[left]+index[right+1])>>1;
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


// Определение номера карзины 
// Формируется положмтельное число из len бит с позиции index
template<class T>
__device__ int device_indexator(T *x, int index, int len)
{
	assert(index+len <= sizeof(T));
	return (int)((((*x) >> index) + (1 << (8 * sizeof(T)-index))) & ((1 << len) - 1));
}

// Функция сравнения данных xранимых в памяти как целых чисел типа long
// comparison function which returns ​a negative integer value if the first argument is less than the second, 
// a positive integer value if the first argument is greater than the second and zero if the arguments are equal.
template<class T>
__device__ int device_comparer(T *x, T *y)
{
	if ((*x)<(*y)) return -1;
	else if ((*x)>(*y)) return 1;
	else return 0;
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
				if (value > 0) device_exchange<T>(&data[i],&data[j],1);
			}
		}
	} else {
		for(int i = 0 ; i < ((index+len) % n) ; i++ ) {
			for(int j = i + 1 ; j <= ((index+len)%n) ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value > 0) device_exchange<T>(&data[i],&data[j],1);
			}
			for(int j = index ; j < n ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value > 0) device_exchange<T>(&data[i],&data[j],1);
			}
		}
		for(int i = index ; i < n-1 ; i++ ) {
			for(int j = i + 1 ; j < n ; j++ ) {
				int value = direction*fn_comparer(&data[i],&data[j]);
				if (value > 0) device_exchange<T>(&data[i],&data[j],1);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	int n;
	char *inputFileName;
	char *outputFileName;

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

	if (argc < 2){
		printf("Usage :\t%s <inputfile> <outputfile>\n", argv[0]); fflush(stdout);
		exit(-1);
	}

	// Получаем параметры - имена файлов
	inputFileName = argv[1];
	outputFileName = argv[2];

	// Подсчитываем количество элементов в файле
	{
		FILE *fl = fopen(inputFileName, "r");
		n = 0;
		long v;
		while (fscanf(fl, "%ld", &v) == 1) n++;
		fclose(fl);
	}

	// Создаём массив длины n чисел типа long
	long *arr = (long *)malloc(n*sizeof(long));

	{
		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Array size :\t%d\n", n);
		printf("Input file name :\t%s\n", inputFileName);
		printf("Output file name :\t%s\n", outputFileName);

		/* Заполняем массив числами */
		/* Операция выполняется только на ведущем процессе */
		FILE *fl = fopen(inputFileName, "r");
		for (int i = 0; i<n; i++) {
			fscanf(fl, "%ld", &arr[i]);
		}
		fclose(fl);
	}

	// Сортируем массив по возрастанию
	fn_parallel_sort(arr, n, 1);

	/* Проверяем и выводим результаты */
	{
		bool check = true;
		for (int i = 0; (i < (n - 1)) && check; i++)
			check = (arr[i] <= arr[i + 1]);

		printf("Array size :\t%d\n", n);
		printf("Check :\t%s\n", (check ? "ok" : "fail"));

		FILE *fl = fopen(outputFileName, "w");
		for (int i = 0; i<n; i++) {
			fprintf(fl, "%ld\n", arr[i]);
		}
		fclose(fl);
	}

	// Высвобождаем массив
	free(arr);

	cudaDeviceReset();
	exit(0);
}