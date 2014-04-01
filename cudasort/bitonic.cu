char *title = "bitonic sort";
char *description = "Битоническая сортировка (bitonic sort)";
/*
В основе этой сортировки лежит операция Bn(полуочиститель, half - cleaner) над массивом, параллельно
упорядочивающая элементы пар xi и xi + n / 2.На рис. 1 полуочиститель может упорядочивать элементы пар как по
возрастанию, так и по убыванию.Сортировка основана на понятии битонической последовательности и
утверждении : если набор полуочистителей правильно сортирует произвольную последовательность нулей и
единиц, то он корректно сортирует произвольную последовательность.
Последовательность a0, a1, …, an - 1 называется битонической, если она или состоит из двух монотонных
частей(т.е.либо сначала возрастает, а потом убывает, либо наоборот), или получена путем циклического
сдвига из такой последовательности.Так, последовательность 5, 7, 6, 4, 2, 1, 3 битоническая, поскольку
получена из 1, 3, 5, 7, 6, 4, 2 путем циклического сдвига влево на два элемента.
Доказано, что если применить полуочиститель Bn к битонической последовательности a0, a1, …, an - 1,
то получившаяся последовательность обладает следующими свойствами :
• обе ее половины также будут битоническими.
• любой элемент первой половины будет не больше любого элемента второй половины.
• хотя бы одна из половин является монотонной.
Применив к битонической последовательности a0, a1, …, an - 1 полуочиститель Bn, получим две
последовательности длиной n / 2, каждая из которых будет битонической, а каждый элемент первой не превысит
каждый элемент второй.Далее применим к каждой из получившихся половин полуочиститель Bn / 2.Получим
уже четыре битонические последовательности длины n / 4.Применим к каждой из них полуочиститель Bn / 2 и
продолжим этот процесс до тех пор, пока не придем к n / 2 последовательностей из двух элементов.Применив к
каждой из них полуочиститель B2, отсортируем эти последовательности.Поскольку все последовательности
уже упорядочены, то, объединив их, получим отсортированную последовательность.
Итак, последовательное применение полуочистителей Bn, Bn / 2, …, B2 сортирует произвольную
битоническую последовательность.Эту операцию называют битоническим слиянием и обозначают Mn.
Например, к последовательности из 8 элементов a 0, a1, …, a7 применим полуочиститель B2, чтобы на
соседних парах порядок сортировки был противоположен.На рис. 2 видно, что первые четыре элемента
получившейся последовательности образуют битоническую последовательность.Аналогично последние
четыре элемента также образуют битоническую последовательность.Поэтому каждую из этих половин можно
отсортировать битоническим слиянием, однако проведем слияние таким образом, чтобы направление
сортировки в половинах было противоположным.В результате обе половины образуют вместе битоническую
Битоническая сортировка последовательности из n элементов разбивается пополам и каждая из
половин сортируется в своем направлении.После этого полученная битоническая последовательность
сортируется битоническим слиянием.
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
// Настроечные аттрибуты
// _comparer - функция сравнения двух элементов массива
// _indexator - функция определения номера корзины для элемента массива
// _non_parallel_sort - фунция сортировки без использования паралельных вычислений
// _parallel_sort - фунция сортировки с использованием паралельных вычислений

#define fn_comparer  device_comparer<long>
#define fn_indexator device_indexator<long>
#define fn_non_parallel_sort device_bubble_sort<long>
#define fn_parallel_sort host_bitonic_sort<long>

template<class T>
__host__ void host_bitonic_sort(T *data, int n, int direction)
{
	// data - массив данных
	// n - количество элементов в исходном массиве для сортировки
	// direction - способ сортировки 
	// -1 означает сортировку по убыванию, 
	//  1 означает сортировку по возрастанию

	cudaError_t err;
	T *device_data;
	T *device_data2;
	int *device_size;

	// Всего надо выполнить k*(k-1)/2*2^(k-1) операций сравнения, где k = log2 n
	// За одну итерацию запуска процессов будет выполнено n/2 = 2^(k-1) операций

	// Шаг первый - копируем исходный массив в память GPU 

	err = cudaMalloc((void**)&device_data, n*sizeof(T));
	cudaMemcpy(device_data, data, n*sizeof(T), cudaMemcpyHostToDevice);

	// Число n представимо в виде суммы степеней двойки,
	// Поэтому, разбиваем исходные данные на подмассивы с длинами равными слагаемым этой суммы
	// и сортируем каждый подмассив битоническим алгоритмом 
	// В разультате получим равное числу слагаеммых отсортированных массивов длинами равным степеням двойки

	for(int k=1; (1<<k) <= n ; k++) {

		if ( n & (1<<k) ) {

			for(int i = 0; i < k ; i++ ) {
				for( int j = i; j >= 0 ; j-- ) 	{ 

					// Определим оптимальное разбиения на процессы, нити и циклы 
					// одна нить в просессе будет будет выполнять цикл с указанным количеством итераций

					int blocks = 1 << (max(1,(int)k/3));
					int threads = 1 << (max(1,(int)k/3));
					int loops = 1 << (k-2*max(1,(int)k/3)-1);

					assert((1<<k) == 2*blocks*threads*loops);

					// одинаковый шаг в каждом блоке гарантирует отсутствие коллизий (одновременного доступа к одним и тем же данным)
					global_bitonic_worker<T> <<< blocks, threads >>>(&device_data[n&((1<<k)-1)], n&(1<<k), i, j, loops, direction);
				}
			}
		}
	}

	// Теперь надо произвести слияние уже отсортированных массивов
	// Для этого выделяет массив такого же размера как и первый
	// и массив размеров очередей

	err = cudaMalloc((void**)&device_data2, n*sizeof(T));
	err = cudaMalloc((void**)&device_size, sizeof(int)*sizeof(int)*8);

	global_bitonic_merger<T> <<< 1, 1 >>>(device_data, device_data2, device_size, n , direction);

	// Возвращаем результаты в исходный массив
	cudaMemcpy(data, device_data2, n*sizeof(T), cudaMemcpyDeviceToHost);

	// Освобождаем память на устройстве
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
					&data[(n&((1<<i)-1))+size[i]-1]) < 0)
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
	// Получаем идентификатор нити
	int block = blockDim.x*blockIdx.x + threadIdx.x;
	int step = 1<<j;
	for(int y=0; y<loops; y++) {
		// Получаем идентификатор шага цикла
		int id = block*loops+y;
		int offset = ((id>>j)<<(j+1))+(id&((1<<j)-1));
		int parity = (id >> i);
		while(parity>1) parity = (parity>>1) ^ (parity&1);
		parity = 1-(parity<<1); // теперь переменная parity может иметь только 2 значения 1 и -1

		assert ((offset+step) < n) ;
		
		int value = parity*direction*fn_comparer(&data[offset],&data[offset+step]);
		if (value > 0) device_exchange<T>(&data[offset],&data[offset+step],1);
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