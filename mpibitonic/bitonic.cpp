#define _CRT_SECURE_NO_WARNINGS

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

#include <stdio.h>
#include <stdlib.h> 
#include <mpi.h> 

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define DATA_TAG 1

/* Фунция сравнения двух элементов для определения порядка сортировки */ 
/* Возвращает <0 если e1 предшедствует e2, 0 если елементы равны и >0 если e2 предшедствует e1 */
// comparison function which returns ​a negative integer value if the first argument is less than the second, 
// a positive integer value if the first argument is greater than the second and zero if the arguments are equal.
int asc_order(const void *e1, const void *e2) 
{ 
	if(*((long *)e1) < *((long *)e2)) return -1; 
	if(*((long *)e1) > *((long *)e2)) return 1; 
	return 0;
}
int desc_order(const void *e1, const void *e2) 
{ 
	if(*((long *)e1) < *((long *)e2)) return 1; 
	if(*((long *)e1) > *((long *)e2)) return -1; 
	return 0;
}

/* Перестановка двух элементов в массиве */
void exchange(const void *e1, const void *e2) 
{ 
	long x = *((long *)e1) ; *((long *)e1) = *((long *)e2); *((long *)e2) = x;
}
/* Копирование элементов */
void copy(const void *e1, const void *e2, int count) 
{ 
	int i;
	for(i = 0 ; i < count ; i++) {
		((long *)e1)[i] = ((long *)e2)[i];
	}
}

/* Bn(полуочиститель, half - cleaner) над массивом, параллельно
упорядочивающая элементы пар xi и xi + half */
void Bn (long *elements, int k, int direction) {
	int i;
	for(i = 0; i<(1<<k) ; i++) {
		if (direction*asc_order(&elements[i],&elements[i+(1<<k)]) > 0) {
			exchange(&elements[i],&elements[i+(1<<k)]);
		}
	}
}
/* последовательное применение полуочистителей Bn, Bn / 2, …, B2 сортирует произвольную
битоническую последовательность.Эту операцию называют битоническим слиянием и обозначают Mn */
void Mn(long *elements, int k, int direction,int myrank,int nrank,int shift) {
	MPI_Status status; 
	Bn(elements, k, direction);
	int child = myrank + shift;
	if (k>0 && child < nrank) {
		/* Запрашиваем первый свободный процесс */
		/* Поскольку здесь не реализован диспечер задач, то это будет следующий по номеру */
		shift <<= 1;
		/* Отдаём половину массива на обработку этому процессу  */
		MPI_Send(&k, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		MPI_Send(&direction, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD);
		MPI_Send(&elements[1<<k], 1<<k, MPI_LONG, child, DATA_TAG, MPI_COMM_WORLD); 
		MPI_Send(&shift, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 

		/* Сами продолжим обработку */
		Mn(elements,k-1,direction,myrank,nrank,shift);

		/* Получим обработанные элементы обратно */
		MPI_Recv(&elements[1<<k], 1<<k, MPI_LONG, child, DATA_TAG, MPI_COMM_WORLD, &status);
		shift >>= 1;
	} else if (k>0) {
		/* Обрабатываем всё сами */
		Mn(elements,k-1,direction,myrank,nrank,shift);
		Mn(&elements[1<<k],k-1,direction,myrank,nrank,shift);
	}
}

int main(int argc, char *argv[]) 
{ 
	int n;         /* Размер сортируемого массива */ 
	int nrank;      /* Общее количество процессов */ 
	int myrank;    /* Номер текущего процесса */ 
	long *elements[2];   /* Массив элементов, хранимые локально */ 
	int *size;   /* Вспомогательный массив */ 
	int direction; /* Порядок сортировки 1 - по возрастанию, -1 по убыванию */
	int i, j, k; 
	MPI_Status status;
	char *inputFileName;
	char *outputFileName;

	/* Иницилизация MPI */ 
	MPI_Init(&argc, &argv); 
	MPI_Comm_size(MPI_COMM_WORLD, &nrank); 
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 

	if (myrank == 0 && argc < 2){
		printf("Usage :\t%s <inputfile> <outputfile>\n", argv[0]); fflush(stdout);
	}

	if (argc < 2){
		MPI_Finalize();
		exit(-1);
	}

	// Получаем параметры - имена файлов
	inputFileName = argv[1];
	outputFileName = argv[2];

	// Подсчитываем количество элементов в файле
	if (myrank == 0){
		FILE *fl = fopen(inputFileName, "r");
		n = 0;
		long v;
		while (fscanf(fl,"%ld", &v)==1) n++;
		fclose(fl);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Устанавливаем сортировку по-возрастанию
	direction = 1;

	elements[0]  = (long *)malloc(n*sizeof(long)); 
	elements[1]  = (long *)malloc(n*sizeof(long)); 
	size  = (int *)malloc(8*sizeof(int)*sizeof(int)); 

	if (myrank == 0) {

		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Number of processes :\t%d\n", nrank);
		printf("Array size :\t%d\n", n);
		printf("Input file name :\t%s\n", inputFileName);
		printf("Output file name :\t%s\n", outputFileName);

		/* Заполняем массив числами */ 
		/* Операция выполняется только на ведущем процессе */ 
		FILE *fl = fopen(inputFileName, "r");
		for (i = 0; i<n; i++) {
			fscanf(fl,"%ld",&elements[0][i]); 
		}
		fclose(fl);

		// Число n представимо в виде суммы степеней двойки,
		// Поэтому, разбиваем исходные данные на подмассивы с длинами равными слагаемым этой суммы
		// и сортируем каждый подмассив битоническим алгоритмом 
		// В разультате получим равное числу слагаеммых отсортированных массивов длинами равным степеням двойки

		for(k = 1; (1<<k) <= n; k++ ) {
			if ((1<<k)&n) {
				int offset = n&((1<<k)-1);
				for(i = 0; i < k ; i++ ) { 
					for(j = 0; j<(1<<(k-i-1)) ; j++) {
						int parity = j; while(parity>1) parity = ((parity>>1) ^ (parity&1));
						parity = 1-(parity<<1); // теперь переменная parity может иметь только 2 значения 1 и -1
						Mn(&elements[0][offset+(j<<(i+1))],i,parity*direction,myrank,nrank,1);
					}
				}
			}
		}

		// Теперь надо произвести слияние уже отсортированных массивов

		for(k=0; k<8*sizeof(int) ; k++ ) size[k] = n & (1<<k);

		int total = n;

		while(total > 0) {
			k = 8*sizeof(int); while( (k-->0) && (size[k] == 0) ) ;
			for (i = k; i-- ; ) {
				if (size[i] > 0 &&	
					direction*asc_order(
					&elements[0][(n&((1<<k)-1))+size[k]-1],
					&elements[0][(n&((1<<i)-1))+size[i]-1]) < 0)
				{
					k = i;
				}
			}
			copy(&elements[1][--total],&elements[0][(n&((1<<k)-1))+(--size[k])],1);
		}

		/* Завершаем все ведомые процессы */

		int stop=-1;
		for(i=1; i<nrank;i++) {
			MPI_Send(&stop, 1, MPI_INT, i, DATA_TAG, MPI_COMM_WORLD); 
		}
	} else {
		while (1==1)
		{
			/* Все ведомые процессы переходят в режим ожидания получения задания от основного процесса */
			// printf("Process %d are waiting for a task\n", myrank);

			/* Получает массив к которому надо применить битоническое слияние */
			MPI_Recv(&k, 1, MPI_INT, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, &status);
			if (k == -1) {
				/* -1 означает сигнал об окончании работы ведомого процесса */
				// printf("Process %d terminated\n", myrank);
				break; /* Выходим из цикла ожидания */
			} else {
				int parent = status.MPI_SOURCE;
				int shift = 0;
				MPI_Recv(&direction, 1, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				MPI_Recv(elements[0], 1<<k, MPI_LONG, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				MPI_Recv(&shift, 1, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				//printf("Process %d has recieved a task from process %d for array of %d items.\n", myrank, parent, 1<<k);
				Mn(elements[0], k-1, direction, myrank, nrank, shift);
				MPI_Send(elements[0], 1<<k, MPI_LONG, parent, DATA_TAG, MPI_COMM_WORLD); 
			}
		}
	}


	/* Проверяем и выводим результаты */
	if (myrank == 0) {
		int check = 1;
		for(i=1; i<n && check==1 ;i++) {
			check = (direction*asc_order(&elements[1][i-1],&elements[1][i]) <= 0)?1:0;
		}

		printf("Check :\t%s\n", (check?"ok":"fail"));

		FILE *fl = fopen(outputFileName, "w");
		for (i = 0; i<n; i++) {
			fprintf(fl, "%ld\n", elements[1][i]);
		}
		fclose(fl);
	}

	/* Освобождаем ранее выделенные ресурсы */
	free(size); 
	free(elements[1]); 
	free(elements[0]); 

	MPI_Finalize(); 
	exit(0);
} 
