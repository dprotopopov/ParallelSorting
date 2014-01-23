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
#define READY_TAG 2

/* Фунция сравнения двух элементов для определения порядка сортировки */ 
/* Возвращает <0 если e1 предшедствует e2, 0 если елементы равны и >0 если e2 предшедствует e1 */
int asc_order(const void *e1, const void *e2) 
{ 
	return (*((int *)e1) - *((int *)e2)); 
}
int desc_order(const void *e1, const void *e2) 
{ 
	return (*((int *)e2) - *((int *)e1)); 
}

/* Перестановка двух элкментов в массиве */
void exchange(const void *e1, const void *e2) 
{ 
	int x = *((int *)e1) ; *((int *)e2) = *((int *)e1); *((int *)e1) = x;
}

/* Bn(полуочиститель, half - cleaner) над массивом, параллельно
упорядочивающая элементы пар xi и xi + half */
void Bn (int *elements, int half, int direction) {
	int i;
	for(i = 0; i<half ; i++) {
		if (direction*asc_order(&elements[i],&elements[i+half]) > 0) {
			exchange(&elements[i],&elements[i+half]);
		}
	}
}
/* последовательное применение полуочистителей Bn, Bn / 2, …, B2 сортирует произвольную
битоническую последовательность.Эту операцию называют битоническим слиянием и обозначают Mn */
void Mn(int *elements, int half, int direction,int myrank,int nrank) {
	MPI_Status status; 
	Bn(elements, half, direction);
	if (half<2) return;
	if ((myrank+1) < nrank) {
		/* Запрашиваем первый свободный процесс */
		int child = 0;
		MPI_Recv(&child, 1, MPI_INT, MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD, &status);
		/* Отдаём половину массива на обработку первому свободному процессу  */
		MPI_Send(&half, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		MPI_Send(&direction, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD);
		MPI_Send(&elements[half], half, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		Mn(elements,half>>1,direction,myrank,nrank);
		MPI_Recv(&elements[half], half, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD, &status);
	} else {
		/* Обрабатываем всё сами */
		Mn(elements,half>>1,direction,myrank,nrank);
		Mn(&elements[half>>1],half>>1,direction,myrank,nrank);
	}
}

main(int argc, char *argv[]) 
{ 
	int n;         /* Размер сортируемого массива */ 
	int nrank;      /* Общее количество процессов */ 
	int myrank;    /* Номер текущего процесса */ 
	int *elements;   /* Массив элементов, хранимые локально */ 
	int *elements2;   /* Массив с результатами сортировки */ 
	int *size;   /* Вспомогательный массив */ 
	int direction; /* Порядок сортировки 1 - по возрастанию, -1 по убыванию */
	int i, j, k; 
	int start, end;
	MPI_Status status; 

	/* Иницилизация MPI */ 
	MPI_Init(&argc, &argv); 
	MPI_Comm_size(MPI_COMM_WORLD, &nrank); 
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 

	n = atoi(argv[1]); 
	direction = 1;
	elements  = (int *)malloc(n*sizeof(int)); 
	elements2  = (int *)malloc(n*sizeof(int)); 
	size  = (int *)malloc(8*sizeof(int)*sizeof(int)); 

	if (myrank == 0) {

		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Number of processes :\t%d\n", nrank);
		printf("Array size :\t%d\n", n);

		/* Заполняем массив псевдо-случайными числами */ 
		/* Операция выполняетя только на ведущем процессе */ 
		for (i=0; i<n; i++) {
			elements[i] = random(); 
		}

		// Число n представимо в виде суммы степеней двойки,
		// Поэтому, разбиваем исходные данные на подмассивы с длинами равными слагаемым этой суммы
		// и сортируем каждый подмассив битоническим алгоритмом 
		// В разультате получим равное числу слагаеммых отсортированных массивов длинами равным степеням двойки

		for(k = 1; (1<<k) <= n; k++ ) {
			if ((1<<k)&n) {
				int offset = n&((1<<k)-1);
				for(i = 0; i < k ; i++ ) { 
					for(j = 0; j<(1<<(k-i-1)) ; j++) {
						int parity = j; while(parity>0) parity = (parity>>1) ^ (parity&1);
						Mn(&elements[offset+(j<<(k-i))],(1<<i),(parity?-direction:direction),myrank,nrank);
					}
				}
			}
		}

		// Теперь надо произвести слияние уже отсортированных массивов
		// Для этого выделяем массив такого же размера как и первый
		// и массив размеров очередей
		for(k=0; k<8*sizeof(int) ; k++ ) size[k] = n & (1<<k);

		int total = n;

		while(total > 0) {
			k = 8*sizeof(int); while( (k-->0) && (size[k] == 0) ) ;
			for (i = k; i-- ; ) {
				if (size[i] > 0 &&	
					direction*asc_order(
						&elements[(n&((1<<k)-1))+size[k]-1],
						&elements[(n&((1<<i)-1))+size[i]-1]) > 0)
				{
					k = i;
				}
			}
			total--;
			size[k]--;
			elements2[total]=elements[(n&((1<<k)-1))+size[k]];
		}

		/* Завершаем все ведомые процессы */
		for(i=1; i<nrank;i++) {
			int zero=0;
			int child = 0;
			MPI_Recv(&child, 1, MPI_INT, MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD, &status);
			MPI_Send(&zero, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		}

	} else {
		while (1==1)
		{
			/* Все ведомые процессы переходят в режим ожидания получения задания от ведущего процесса */
			/* Получают массив к которому надо применить битоническое слияние */
			MPI_Send(&myrank, 1, MPI_INT, MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD); 
			MPI_Recv(&n, 1, MPI_INT, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, &status);
			if (n == 0) {
				/* Получен сигнал об окончании работы алгоритма */
				break; /* Выходим из цикла ожидания */
			} else {
				int parent = status.MPI_SOURCE ;
				MPI_Recv(&direction, 1, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				MPI_Recv(elements, n, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				Mn(elements, n>>1, direction, myrank, nrank);
				MPI_Send(elements, n, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD); 
			}
		}
	}


	/* Проверяем и выводим результаты */
	if (myrank == 0) {
		int check = 1;
		for(i=1; i<n && check ;i++) {
			check = (direction*asc_order(&elements2[i-1],&elements2[i]) <= 0)?1:0;
		}

		printf("Check :\t%s\n", (check?"ok":"fail"));

		for(i=0; i<n && i<20;i++) {	printf("%d,",elements2[i]); } printf("...\n");
	} 

	/* Освобождаем ранее выделенные ресурсы */
	free(size); 
	free(elements2); 
	free(elements); 

	MPI_Finalize(); 

	exit(0);
} 
