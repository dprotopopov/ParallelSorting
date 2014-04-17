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

#include <stdio.h>
#include <stdlib.h> 
#include <mpi.h> 

#define DATA_TAG 1

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

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
	int x = *((int *)e1); *((int *)e1) = *((int *)e2); *((int *)e2) = x;
}
/* Копирование элементов */
void copy(const void *e1, const void *e2, int count)
{
	int i;
	for (i = 0; i < count; i++) {
		((int *)e1)[i] = ((int *)e2)[i];
	}
}

int main(int argc, char *argv[])
{
	int n;         /* Размер сортируемого массива */
	int nrank;      /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */
	int maxnlocal;    /* Максимальное количество элементов, хранимых локально */
	int nlocal;    /* Количество элементов, хранимых локально */
	int nremote;    /* Количество элементов, полученных при операции обмена от соседа */
	int *elements[2];   /* Массив элементов, хранимые локально */
	int oddrank;   /* Номер соседа при нечётной фазе обмена */
	int evenrank;  /* Номер соседа при чётной фазе обмена */
	int direction; /* Порядок сортировки 1 - по возрастанию, -1 по убыванию */
	int i, j;
	int start, end;
	MPI_Status status;

	/* Иницилизация MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if (myrank == 0 && argc < 2){
		printf("Usage :\t%s <arraysize>\n", argv[0]); fflush(stdout);
	}

	if (argc < 2){
		MPI_Finalize();
		exit(-1);
	}

	n = atoi(argv[1]);
	direction = 1;

	if (myrank == 0) {
		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Number of processes :\t%d\n", nrank);
		printf("Array size :\t%d\n", n);
	}

	maxnlocal = (int)((n + nrank - 1) / nrank); /* Вычисляем максимальное количество элементов хранимых одним процессом */

	/* Выделяем память под хранение данных */
	if (myrank == 0) {
		elements[0] = (int *)malloc(max(n*sizeof(int), 2 * maxnlocal*sizeof(int)));
		elements[1] = (int *)malloc(max(n*sizeof(int), 2 * maxnlocal*sizeof(int)));
	}
	else {
		elements[0] = (int *)malloc(2 * maxnlocal*sizeof(int));
		elements[1] = (int *)malloc(2 * maxnlocal*sizeof(int));
	}

	if (elements) {
		/* Заполняем массив псевдо-случайными числами */
		/* Операция выполняетя только на ведущем процессе */
		if (myrank == 0) {
			for (i = 0; i < n; i++) {
				elements[0][i] = rand();
			}
		}
	}

	/* Распределям данные между процессами */
	/* Начальные данные лежат в массиве elements главного процесса */
	/* Количество данных полученных одним прцессом может различатся */
	/* поскольку размер исходных данных может быть не кратным количеиву процессов */
	if (myrank == 0) {
		nlocal = n;
		for (i = nrank; i > 1; i--) {
			nremote = (int)nlocal / i;
			nlocal -= nremote;
			MPI_Send(&nremote, 1, MPI_INT, i - 1, DATA_TAG, MPI_COMM_WORLD); /* Уведомляем о количестве передаваемых данных */
			if (nremote > 0) MPI_Send(&elements[0][nlocal], nremote, MPI_INT, i - 1, DATA_TAG, MPI_COMM_WORLD); /* Отправляем данные */
		}
	}
	else {
		MPI_Recv(&nlocal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		if (nlocal > 0) MPI_Recv(elements[0], maxnlocal, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD, &status);
	}

	/* Определяем номера процессов, используемых при операциях обмена с соседом */
	if (myrank % 2 == 0) {
		oddrank = myrank - 1;
		evenrank = myrank + 1;
	}
	else {
		oddrank = myrank + 1;
		evenrank = myrank - 1;
	}

	/* Поскольку мы не можем гарантировать чётное количество процессов, то организуем процессы в линейную схему */
	if (oddrank == -1 || oddrank == nrank)
		oddrank = MPI_PROC_NULL;
	if (evenrank == -1 || evenrank == nrank)
		evenrank = MPI_PROC_NULL;

	/* Выполняем предварительный цикл алгоритма */

	qsort(elements[0], nlocal, sizeof(int), ((direction > 0) ? asc_order : desc_order));

	/* Выполняем основной цикл алгоритма */
	for (i = 0; i < nrank; i++) {
		int neigborrank = (i % 2 == 1) ? oddrank : evenrank;
		nremote = 0;
		/* Обмениваемся информацией о количестве данных */
		MPI_Sendrecv(&nlocal, 1, MPI_INT, neigborrank, DATA_TAG, &nremote, 1, MPI_INT, neigborrank, DATA_TAG, MPI_COMM_WORLD, &status);
		/* Обмениваемся данными - полученные данные загружаем в конец локального массива */
		MPI_Sendrecv(elements[i & 1], maxnlocal, MPI_INT, neigborrank, DATA_TAG, &elements[i & 1][nlocal], maxnlocal, MPI_INT, neigborrank, DATA_TAG, MPI_COMM_WORLD, &status);

		// Запускаем алгоритм для слияния отсортированных частей массива
		int total = nlocal + nremote;
		int size0 = nlocal;
		int size1 = nremote;
		int index0 = 0;
		int index1 = nlocal;
		while ((size0 > 0) && (size1 > 0)) {
			int value = direction*asc_order(&elements[i & 1][index0 + size0 - 1], &elements[i & 1][index1 + size1 - 1]);
			if (value > 0) { copy(&elements[1 - (i & 1)][--total], &elements[i & 1][index0 + (--size0)], 1); }
			else { copy(&elements[1 - (i & 1)][--total], &elements[i & 1][index1 + (--size1)], 1); }
		}
		copy(&elements[1 - (i & 1)][total - size0], &elements[i & 1][index0], size0);
		copy(&elements[1 - (i & 1)][total - size0 - size1], &elements[i & 1][index1], size1);

		/* Оставляем у себя только левую или правую часть в зависимости от чётности шага и номера процесса */
		if ((myrank < neigborrank) && (myrank < nrank - 1)) {
			/* Был обмен с процессом справа */
			/* Данные лежат в начале объединёного массива - ничего делать не надо */
		}
		else if ((myrank > neigborrank) && (myrank > 0)) {
			/* Был обмен с процессом слева */
			/* Данные лежат в конце объединёного массива - перемещаем их в начало массива */
			copy(&elements[1 - (i & 1)][0], &elements[1 - (i & 1)][nremote], nlocal);
		}
	}

	/* Собираем данные с ведомых процессов */
	if (myrank == 0) {
		for (i = 1; i<nrank; i++) {
			nremote = 0;
			MPI_Recv(&nremote, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
			if (nremote > 0) MPI_Recv(&elements[nrank & 1][nlocal], nremote, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
			nlocal += nremote;
		}
	}
	else {
		MPI_Send(&nlocal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); /* Уведомляем о количестве передаваемых данных */
		if (nlocal > 0) MPI_Send(elements[nrank & 1], nlocal, MPI_INT, 0, 1, MPI_COMM_WORLD); /* Отправляем данные */
	}


	/* Проверяем и выводим результаты */
	/* Результаты сортировки лежат в массиве elements[nrank&1] */
	if (myrank == 0) {
		int check = 1;
		for (i = 1; i < n && check; i++) {
			check = (direction*asc_order(&elements[nrank & 1][i - 1], &elements[nrank & 1][i]) <= 0) ? 1 : 0;
		}

		printf("Check :\t%s\n", (check ? "ok" : "fail"));

		for (i = 0; i < n && i < 20; i++) { printf("%d,", elements[nrank & 1][i]); } printf("...\n");
	}

	/* Освобождаем ранее выделенные ресурсы */
	free(elements[1]);
	free(elements[0]);

	MPI_Finalize();
	exit(0);
}


