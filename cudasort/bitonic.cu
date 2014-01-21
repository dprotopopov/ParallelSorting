/*
Битоническая сортировка(bitonic sort)
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

__host__ void host_bitonic_sort(byte *data, int n, size_t size, int direction)
{
// data - массив данных
// n - количество элементов в исходном массиве для сортировки
// size - размер одного элемента массива в байтах
// direction - способ сортировки 
// -1 означает сортировку по убыванию, 
//  1 означает сортировку по возрастанию
	
	cudaError_t err;
	byte *device_data;

	// Всего надо выполнить k*(k-1)/2*2^(k-1) операций сравнения, где k = log2 n
	// За одну итерацию запуска процессов будет выполнено n/2 = 2^(k-1) операций

	// Определим оптимальное разбиения на процессы, нити и циклы 
	// одна нить в просессе будет будет выполнять циккл с указанным количеством итераций

	int blocks = min(max(1,(int)pow((double)n,0.33333333333)),255);
	int threads = max(1,(int)sqrt((double)n/blocks));
	int loops = (int)(n+2*blocks*threads-1)/(2*blocks*threads);

	assert(n <= 2*blocks*threads*loops);

	// Шаг первый - копируем исходный массив в память GPU 

	error = cudaMalloc((void**)&device_data, n*size);
	cudaMemcpy(device_data, data, n*size, cudaMemcpyHostToDevice);

	int i = 0;
	do {
		i++; // логарифм - 1 размера блока
		for( int j = i; j-- > 0 ; ) // логарифм размера шага
		{ 
			// одинаковый шаг в каждом блоке гарантирует отсутствие коллизий (одновременного доступа к одним и тем же данным)
			global_bitonic_worker <<< blocks, threads >>>(device_data, n, i, j, loops, size, direction);
		}
	}
	while( (1<<i) < n );

	// Возвращаем результаты в исходный массив
	cudaMemcpy(data, device_data, n*size, cudaMemcpyDeviceToHost);

	// Освобождаем память на устройстве
	cudaFree(device_data);
}

__global__ void global_bitonic_worker(
	bype * data, 
	int n, int i, int j,
	int loops,
	size_t size,
	int direction)
{
	// Получаем идентификатор нити
	int block = blockDim.x*blockIdx.x + threadIdx.x;
	for(int y=0; y<loops; y++) {
		// Получаем идентификатор шага цикла
		int id = block*loops+y;
		int step = 1<<j;
		int offset = ((id>>j)<<(j+1))+(id&((1<<j)-1);
		if ((offset+step) < n) {
			int parity = (id>>i);
			while(parity>1) parity = (parity>>1) ^ (parity&1);
			parity = (parity<<1)-1; // теперь переменная parity может иметь только 2 значения 1 и -1
			int value = parity*direction*(comparer)(&data[offset*size],&data[(offset+step)*size],size);
			if (value < 0) device_exchange(&data[index*size],&data[(index+step)*size],size);
		}
	}
}

// Перестановка двух блоков в памяти устройства
__device__ void device_exchange(byte *x, byte *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		byte ch = x[i] ; x[i] = y[i] ; y[i] = ch;
	}
}

// Копирование одного участка памяти в другой
__device__ void device_copy(byte *x, byte *y, int count)
{
	for(int i = 0; i < count ; i++ ) {
		x[i] = y[i] ;
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
// Формируется положмтельное число из len бит с позиции index
__device__ int device_integer_indexator(byte *x, int index, int len, size_t size)
{
	assert(size == sizeof(long));
	return min(m, (((*(long*)x) >> index) + (1 << (8 * sizeof(long)-index))) ^ ((1 << len) - 1);
}

/////////////////////////////////////////////////////////////////
// Битоническая сортировка
__device__ void device_bitonic_sort(byte *arr, int index, int len, int n, size_t size, int direction)
{
	assert(index+len < n);
	assert(len > 0);
	int i = 0;
	do {
		i++; // логарифм - 1 размера блока 
		for( int j = i; j-- > 0 ; ) // логарифм размера шага
		{ 
			for(int id=0; (2*id) < len; id++)
			{
				int step = 1<<j;
				int offset = ((id>>j)<<(j+1))+(id&((1<<j)-1);
				if ((offset+step) < len) {
					int parity = (id>>i);
					while(parity>1) parity = (parity>>1) ^ (parity&1);
					parity = (parity<<1)-1; // теперь переменная parity может иметь только 2 значения 1 и -1
					int value = parity*direction*(comparer)(&data[offset*size],&data[((offset+step)%n)*size],size);
					if (value < 0) device_exchange(&data[offset*size],&data[((offset+step)%n)*size],size);
				}
			}
		}
	}
	while( (1<<i) < len );
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
	_non_parallel_sort = device_bitonic_sort;
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