#define _CRT_SECURE_NO_WARNINGS

char *title = "odd-even sort";
char *description = "���������� �����-��������� �������������� (odd-even sort)";
/*
��� ������ �������� ��������� �������� ���������-������ ��� ���� ��� ��������� ���������� �
����������� ������������. ���������� ������, ����� ����� ����������� ����� ����� ���������, �.�. p=n -
����� ����������� (����������� ���������). �����������, ��� �������������� ������� ����� ���������
������. ����� �������� ai (i = 1, .. , n), ������������� ����������� �� ����������� pi (i = 1, ... , n). � ��������
�������� ������ ��������� � �������� ������� ���������� ���������-����� ������ �������� � ���������,
����������� �� ����������-������ ������. ���������� � ������� ������ �������� ������ ��������� � ������
������� ���������� ���������-����� ������ �������� � ��������� ������� ������.
�� ������ �������� ��������� �������� � ������ ���������� ��������� ��� ���������-������ � ��
������� �������� �� ����� Q(1). ����� ���������� ����� �������� � n; ������� ����� ����������
������������ ���������� � Q(n).
����� ����� ����������� p ������ ����� ��������� n, �� ������ �� ��������� �������� ���� ����
������ n/p � ��������� ��� �� ����� Q((n/p)�log(n/p)). ����� ���������� �������� p �������� (�/2 � ������, �
��������) � ������ �����������-���������: ������� ���������� �������� ���� ����� ���� ������, �
��������� �� ��������� (�� ������ ���� ����������� �������� ���������� �������). ����� ���������
������ ������� �� 2 �����; ����� ��������� ������������ ����� ������ ����� ����� (� �������� ����������
������), � ������ � ������ ������ (� �������� ���������� ������). �������� ��������������� ������
����� p ��������
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

/* ������ ��������� ���� ��������� ��� ����������� ������� ���������� */
/* ���������� <0 ���� e1 ������������� e2, 0 ���� �������� ����� � >0 ���� e2 ������������� e1 */
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

/* ������������ ���� ��������� � ������� */
void exchange(const void *e1, const void *e2)
{
	long x = *((long *)e1); *((long *)e1) = *((long *)e2); *((long *)e2) = x;
}
/* ����������� ��������� */
void copy(const void *e1, const void *e2, int count)
{
	int i;
	for (i = 0; i < count; i++) {
		((long *)e1)[i] = ((long *)e2)[i];
	}
}

int main(int argc, char *argv[])
{
	int n;         /* ������ ������������ ������� */
	int nrank;      /* ����� ���������� ��������� */
	int myrank;    /* ����� �������� �������� */
	int maxnlocal;    /* ������������ ���������� ���������, �������� �������� */
	int nlocal;    /* ���������� ���������, �������� �������� */
	int nremote;    /* ���������� ���������, ���������� ��� �������� ������ �� ������ */
	long *elements[2];   /* ������ ���������, �������� �������� */
	int oddrank;   /* ����� ������ ��� �������� ���� ������ */
	int evenrank;  /* ����� ������ ��� ������ ���� ������ */
	int direction; /* ������� ���������� 1 - �� �����������, -1 �� �������� */
	int i;
	MPI_Status status;
	char *inputFileName;
	char *outputFileName;

	/* ������������ MPI */
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

	// �������� ��������� - ����� ������
	inputFileName = argv[1];
	outputFileName = argv[2];

	// ������������ ���������� ��������� � �����
	if (myrank == 0){
		FILE *fl = fopen(inputFileName, "r");
		n = 0;
		while (fscanf(fl, "%d", &i) == 1) n++;
		fclose(fl);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// ������������� ���������� ��-�����������
	direction = 1;

	if (myrank == 0) {
		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Number of processes :\t%d\n", nrank);
		printf("Array size :\t%d\n", n);
		printf("Input file name :\t%s\n", inputFileName);
		printf("Output file name :\t%s\n", outputFileName);
	}

	maxnlocal = (int)((n + nrank - 1) / nrank); /* ��������� ������������ ���������� ��������� �������� ����� ��������� */

	/* �������� ������ ��� �������� ������ */
	if (myrank == 0) {
		elements[0] = (long *)malloc(max(n*sizeof(long), 2 * maxnlocal*sizeof(long)));
		elements[1] = (long *)malloc(max(n*sizeof(long), 2 * maxnlocal*sizeof(long)));
	}
	else {
		elements[0] = (long *)malloc(2 * maxnlocal*sizeof(long));
		elements[1] = (long *)malloc(2 * maxnlocal*sizeof(long));
	}

	if (myrank == 0) {
		/* ��������� ������ ������� */
		/* �������� ����������� ������ �� ������� �������� */
		FILE *fl = fopen(inputFileName, "r");
		for (i = 0; i<n; i++) {
			fscanf(fl, "%ld", &elements[0][i]);
		}
		fclose(fl);
	}

	/* ����������� ������ ����� ���������� */
	/* ��������� ������ ����� � ������� elements �������� �������� */
	/* ���������� ������ ���������� ����� �������� ����� ���������� */
	/* ��������� ������ �������� ������ ����� ���� �� ������� ��������� ��������� */
	if (myrank == 0) {
		nlocal = n;
		for (i = nrank; i > 1; i--) {
			nremote = (int)nlocal / i;
			nlocal -= nremote;
			MPI_Send(&nremote, 1, MPI_INT, i - 1, DATA_TAG, MPI_COMM_WORLD); /* ���������� � ���������� ������������ ������ */
			if (nremote > 0) MPI_Send(&elements[0][nlocal], nremote, MPI_LONG, i - 1, DATA_TAG, MPI_COMM_WORLD); /* ���������� ������ */
		}
	}
	else {
		MPI_Recv(&nlocal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		if (nlocal > 0) MPI_Recv(elements[0], maxnlocal, MPI_LONG, 0, DATA_TAG, MPI_COMM_WORLD, &status);
	}

	/* ���������� ������ ���������, ������������ ��� ��������� ������ � ������� */
	if (myrank % 2 == 0) {
		oddrank = myrank - 1;
		evenrank = myrank + 1;
	}
	else {
		oddrank = myrank + 1;
		evenrank = myrank - 1;
	}

	/* ��������� �� �� ����� ������������� ������ ���������� ���������, �� ���������� �������� � �������� ����� */
	if (oddrank == -1 || oddrank == nrank)
		oddrank = MPI_PROC_NULL;
	if (evenrank == -1 || evenrank == nrank)
		evenrank = MPI_PROC_NULL;

	/* ��������� ��������������� ���� ��������� */

	qsort(elements[0], nlocal, sizeof(int), ((direction > 0) ? asc_order : desc_order));

	/* ��������� �������� ���� ��������� */
	for (i = 0; i < nrank; i++) {
		int neigborrank = (i % 2 == 1) ? oddrank : evenrank;
		nremote = 0;
		/* ������������ ����������� � ���������� ������ */
		MPI_Sendrecv(&nlocal, 1, MPI_INT, neigborrank, DATA_TAG, &nremote, 1, MPI_INT, neigborrank, DATA_TAG, MPI_COMM_WORLD, &status);
		/* ������������ ������� - ���������� ������ ��������� � ����� ���������� ������� */
		MPI_Sendrecv(elements[i & 1], maxnlocal, MPI_LONG, neigborrank, DATA_TAG, &elements[i & 1][nlocal], maxnlocal, MPI_INT, neigborrank, DATA_TAG, MPI_COMM_WORLD, &status);

		// ��������� �������� ��� ������� ��������������� ������ �������
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

		/* ��������� � ���� ������ ����� ��� ������ ����� � ����������� �� �������� ���� � ������ �������� */
		if ((myrank < neigborrank) && (myrank < nrank - 1)) {
			/* ��� ����� � ��������� ������ */
			/* ������ ����� � ������ ����������� ������� - ������ ������ �� ���� */
		}
		else if ((myrank > neigborrank) && (myrank > 0)) {
			/* ��� ����� � ��������� ����� */
			/* ������ ����� � ����� ����������� ������� - ���������� �� � ������ ������� */
			copy(&elements[1 - (i & 1)][0], &elements[1 - (i & 1)][nremote], nlocal);
		}
	}

	/* �������� ������ � ������� ��������� */
	if (myrank == 0) {
		for (i = 1; i<nrank; i++) {
			nremote = 0;
			MPI_Recv(&nremote, 1, MPI_INT, i, DATA_TAG, MPI_COMM_WORLD, &status);
			if (nremote > 0) MPI_Recv(&elements[nrank & 1][nlocal], nremote, MPI_LONG, i, DATA_TAG, MPI_COMM_WORLD, &status);
			nlocal += nremote;
		}
	}
	else {
		MPI_Send(&nlocal, 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD); /* ���������� � ���������� ������������ ������ */
		if (nlocal > 0) MPI_Send(elements[nrank & 1], nlocal, MPI_LONG, 0, DATA_TAG, MPI_COMM_WORLD); /* ���������� ������ */
	}

	/* ��������� � ������� ���������� */
	if (myrank == 0) {
		int check = 1;
		for (i = 1; i<n && check==1; i++) {
			check = (direction*asc_order(&elements[nrank & 1][i - 1], &elements[nrank & 1][i]) <= 0) ? 1 : 0;
		}

		printf("Array size :\t%d\n", n);
		printf("Check :\t%s\n", (check ? "ok" : "fail"));

		FILE *fl = fopen(outputFileName, "w");
		for (i = 0; i<n; i++) {
			fprintf(fl, "%ld\n", elements[nrank & 1][i]);
		}
		fclose(fl);
	}

	/* ����������� ����� ���������� ������� */
	free(elements[1]);
	free(elements[0]);

	MPI_Finalize();
	exit(0);
}


