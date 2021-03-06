﻿#define _CRT_SECURE_NO_WARNINGS

char *title = "bucket sort";
char *description = "Блочная сортировка (bucket sort)";
/*
В блочной карманной или корзинной сортировке(Bucket sort) сортируемые элементы распределены
между конечным числом отдельных блоков(карманов, корзин).Каждый блок затем сортируется отдельно либо
рекурсивно тем же методом либо другим. Затем элементы помещают обратно в массив.
Для этой сортировки характерно линейное время исполнения.
Алгоритм требует знаний о природе сортируемых данных, выходящих за рамки функций "сравнить" и
"поменять местами", достаточных для сортировки слиянием, сортировки пирамидой, быстрой сортировки,
сортировки Шелла, сортировки вставкой.
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

// Определение номера карзины 
// Формируется положмтельное число из len бит с позиции index
int indexator(const void *e1, int index, int len)
{
	return (int)(((*(long*)e1) >> index) + (1 << (len-1)));
}

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

int main(int argc, char *argv[]) 
{ 
	int n;         /* Размер сортируемого массива */ 
	int nrank;      /* Общее количество процессов */ 
	int myrank;    /* Номер текущего процесса */ 
	long *elements;   /* Массив элементов, хранимые локально */ 
	long ** bucket;   /* Массив корзин, хранимые локально */ 
	int *size;   /* Массив размеров корзин */ 
	int *bucketid;   /* Идентификатор корзины */ 
	int direction; /* Порядок сортировки 1 - по возрастанию, -1 по убыванию */
	int len; /* Параметр для индексатора */
	int index; /* Параметр для индексатора */
	int nbucket; /* Количество корзин на одном процессе */ 
	int i, j, k, m, id, x; 
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

	len = 0; while ((1 << (++len))<n); len = (int)((len + 1) / 2);
	index = 8*sizeof(int)-len;
	nbucket = (int)((1<<len)+nrank-1)/nrank;

	elements  = (long *)malloc(n*sizeof(long)); 

	if (myrank == 0) {

		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Number of processes :\t%d\n", nrank);
		printf("Array size :\t%d\n", n);
		printf("Indexator :\t%d %d\n", index, len);
		printf("Number of buckets per process :\t%d\n", nbucket);
		printf("Input file name :\t%s\n", inputFileName);
		printf("Output file name :\t%s\n", outputFileName);

		/* Заполняем массив числами */
		/* Операция выполняется только на ведущем процессе */
		FILE *fl = fopen(inputFileName, "r");
		for (i = 0; i<n; i++) {
			fscanf(fl, "%ld", &elements[i]);
		}
		fclose(fl);
	}

	bucket = (long**)malloc(nbucket*sizeof(long*));
	size = (int*)malloc(nbucket*sizeof(int));
	bucketid = (int*)malloc(nbucket*sizeof(int));

	for(j=0; j<nbucket ; j++) {
		bucket[j] = (long*)malloc(n*sizeof(long));
	}

	nbucket = 0;

	while(n--) {
		if (myrank == 0) {
			x = elements[n];
		}

		MPI_Bcast(&x, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		id = indexator(&x,index,len); /* Определяем номер корзины */
		if (id%nrank == myrank) {
			/* Если корзина обслуживается данным процессом, то процесс добавляет элемент в корзину */
			j = nbucket; while (j-- && bucketid[j]!=id);
			if (j==-1) { size[nbucket]=0; bucketid[nbucket] = id; bucket[nbucket][size[nbucket]++] = x; nbucket++; } 
			else bucket[j][size[j]++] = x;
		}
	}

	/* Сортируем каждую корзину в отдельности */

	for(i=0; i< nbucket ;i++) {
		qsort(bucket[i],size[i],sizeof(long),((direction>0)?asc_order:desc_order));
	}

	/* Собираем корзины вместе в соответсвии с порядком сортировки */

	n=0;

	for(k = 0; k < (1<<len) ; k++) {
		id = (direction>0)?k:((1<<len)-k-1);
		i = id%nrank; /* Определяем номер процесса хранящего корзину */
		j = nbucket; while ((j--) && (bucketid[j]!=id));
		if (myrank == i && myrank == 0 && j != -1) { /* Если корзина на главном процессе */
			copy(&elements[n], bucket[j], size[j]);
			n += size[j];
		} 
		else if (myrank == i && myrank > 0 && j != -1) {
			MPI_Send(&size[j], 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD); /* Уведомляем о количестве передаваемых данных */
			MPI_Send(bucket[j], size[j], MPI_LONG, 0, DATA_TAG, MPI_COMM_WORLD);
		}
		else if (myrank == i && myrank > 0 && j == -1) {
			int zero = 0;
			MPI_Send(&zero, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); /* Уведомляем о количестве передаваемых данных */
		}
		else if (myrank != i && myrank == 0 && nrank > 1) {
			MPI_Recv(&m, 1, MPI_INT, i, DATA_TAG, MPI_COMM_WORLD, &status);
			if (m>0) MPI_Recv(&elements[n], m, MPI_LONG, i, DATA_TAG, MPI_COMM_WORLD, &status);
			n += m;
		}
	}


	/* Проверяем и выводим результаты */
	if (myrank == 0) {
		int check = 1;
		for(i=1; i<n && check==1 ;i++) {
			check = (direction*asc_order(&elements[i-1],&elements[i]) <= 0)?1:0;
		}

		printf("Array size :\t%d\n", n);
		printf("Check :\t%s\n", (check?"ok":"fail"));

		FILE *fl = fopen(outputFileName, "w");
		for (i = 0; i<n; i++) {
			fprintf(fl, "%ld\n", elements[i]);
		}
		fclose(fl);
	}

	/* Освобождаем ранее выделенные ресурсы */

	for(i=0; i<nbucket ; i++) {
		free(bucket[i]);
	}

	free(bucket);
	free(bucketid); 
	free(size); 
	free(elements); 

	MPI_Finalize(); 
	exit(0);
} 
