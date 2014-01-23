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
����� p ��������, �������� � ������ ����� ����:
23)���� ��������� ������� ������ �� ���������;
24)���� ������������ ������ ����� �������;
25)����� ��������� ���� ����� ������;
26)����� ������������ �������, ��� ���� ��������� ������ � �������� �����;
27)���� ���������� ������������ ����� ���������� ������, �� �������� ��������;
28)���� ��������� ��������������� ������.
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

/* ������ ��������� ���� ��������� ��� ����������� ������� ���������� */ 
/* ���������� <0 ���� e1 ������������� e2, 0 ���� �������� ����� � >0 ���� e2 ������������� e1 */
int asc_order(const void *e1, const void *e2) 
{ 
	return (*((int *)e1) - *((int *)e2)); 
}
int desc_order(const void *e1, const void *e2) 
{ 
	return (*((int *)e2) - *((int *)e1)); 
}

main(int argc, char *argv[]) 
{ 
	int n;         /* ������ ������������ ������� */ 
	int nrank;      /* ����� ���������� ��������� */ 
	int myrank;    /* ����� �������� �������� */ 
	int maxnlocal;    /* ������������ ���������� ���������, �������� �������� */ 
	int nlocal;    /* ���������� ���������, �������� �������� */ 
	int nremote;    /* ���������� ���������, ���������� ��� �������� ������ �� ������ */ 
	int *elements;   /* ������ ���������, �������� �������� */ 
	int oddrank;   /* ����� ������ ��� �������� ���� ������ */ 
	int evenrank;  /* ����� ������ ��� ������ ���� ������ */ 
	int direction; /* ������� ���������� 1 - �� �����������, -1 �� �������� */
	int i, j; 
	int start, end;
	MPI_Status status; 

	/* ������������ MPI */ 
	MPI_Init(&argc, &argv); 
	MPI_Comm_size(MPI_COMM_WORLD, &nrank); 
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 

	n = atoi(argv[1]); 
	direction = 1;

	if (myrank == 0) {
		printf("Title :\t%s\n", title);
		printf("Description :\t%s\n", description);
		printf("Number of processes :\t%d\n", nrank);
		printf("Array size :\t%d\n", n);
	} 

	maxnlocal = (int)((n+nrank-1)/nrank); /* ��������� ������������ ���������� ��������� �������� ����� ��������� */ 

	/* �������� ������ ��� �������� ������ */ 
	if (myrank == 0) {
		elements  = (int *)malloc(max(n*sizeof(int),2*maxnlocal*sizeof(int))); 
	} else {
		elements  = (int *)malloc(2*maxnlocal*sizeof(int)); 
	}
	
	if (elements) {
		/* ��������� ������ ������-���������� ������� */ 
		/* �������� ���������� ������ �� ������� �������� */ 
		if (myrank == 0) {
			for (i=0; i<n; i++) {
				elements[i] = random(); 
			}
		}
	} else {
		printf("Prosess %d malloc error\n", myrank);
	}

	/* ����������� ������ ����� ���������� */
	/* ���������� ������ ���������� ����� �������� ����� ���������� */
	/* ��������� ������ �������� ������ ����� ���� �� ������� ��������� ��������� */
	if (myrank == 0) {
		nlocal = n;
		for(i=nrank; i>1 ; i--) {
			nremote = (int)nlocal/i;
			nlocal -= nremote;
			MPI_Send(&nremote, 1, MPI_INT, i-1, 1, MPI_COMM_WORLD); /* ���������� � ���������� ������������ ������ */
			if(nremote > 0) MPI_Send(&elements[nlocal], nremote, MPI_INT, i-1, 1, MPI_COMM_WORLD); /* ���������� ������ */
		}
	} else {
		MPI_Recv(&nlocal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status); 
		if(nlocal > 0) MPI_Recv(elements, maxnlocal, MPI_INT, 0, 1, MPI_COMM_WORLD, &status); 
	}

	printf("Process %d local array size : %d\n", myrank, nlocal);

	/* ���������� ������ ���������, ������������ ��� ��������� ������ � ������� */ 
	if (myrank%2 == 0) { 
		oddrank  = myrank-1; 
		evenrank = myrank+1; 
	} 
	else { 
		oddrank  = myrank+1; 
		evenrank = myrank-1; 
	} 

	/* ��������� �� �� ����� ������������� ������ ���������� ���������, �� ���������� �������� � �������� ����� */ 
	if (oddrank == -1 || oddrank == nrank) 
		oddrank = MPI_PROC_NULL; 
	if (evenrank == -1 || evenrank == nrank) 
		evenrank = MPI_PROC_NULL; 

	/* ��������� ��������������� ���� ��������� */ 

	qsort(elements, nlocal, sizeof(int), ((direction>0)?asc_order:desc_order)); 

	/* ��������� �������� ���� ��������� */ 
	for (i=0; i<nrank; i++) { 
		int neigborrank = (i%2 == 1)?oddrank:evenrank;
		nremote = 0;
		/* ������������ ����������� � ���������� ������ */
		MPI_Sendrecv(&nlocal, 1, MPI_INT, neigborrank, 1, &nremote, 1, MPI_INT, neigborrank, 1, MPI_COMM_WORLD, &status); 
		/* ������������ ������� - ���������� ������ ��������� � ����� ���������� ������� */
		MPI_Sendrecv(elements, maxnlocal, MPI_INT, neigborrank, 1, &elements[nlocal], maxnlocal, MPI_INT, neigborrank, 1, MPI_COMM_WORLD, &status); 
		
		printf("Process %d iteraction %d recieved %d items from %d\n", myrank, i, nremote, neigborrank);

		/* ��������� ��������� � ���������� �������� */
		/* ��� ��������� ��������� ���������� ������� qsort */
		/* ���� ����� ������������ �������� ������� ���� ��� ��������������� �������� */
		qsort(elements, nlocal+nremote, sizeof(int), ((direction>0)?asc_order:desc_order)); 

		/* ��������� � ���� ������ ����� ��� ������ ����� � ����������� �� �������� ���� � ������ �������� */
		if ( (myrank < neigborrank) && (myrank < nrank-1) ) { 
			/* ��� ����� � ��������� ������ */ 
			/* ������ ����� � ������ ����������� ������� - ������ ������ �� ���� */
		} else if ( (myrank > neigborrank) && (myrank > 0) ) {
			/* ��� ����� � ��������� ����� */ 
			/* ������ ����� � ����� ����������� ������� - ���������� �� � ������ ������� */
			for(j=0;j<nlocal; j++) elements[j] = elements[j+nremote];
		}
	} 

	/* �������� ������ � ������� ��������� */
	if (myrank == 0) {
		for(i=1; i<nrank ; i++) {
			nremote = 0;
			MPI_Recv(&nremote, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status); 
			if(nremote > 0) MPI_Recv(&elements[nlocal], nremote, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
			nlocal += nremote;
		}
	} else {
		MPI_Send(&nlocal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); /* ���������� � ���������� ������������ ������ */
		if(nlocal > 0) MPI_Send(elements, nlocal, MPI_INT, 0, 1, MPI_COMM_WORLD); /* ���������� ������ */			 
	}

	/* ��������� � ������� ���������� */
	if (myrank == 0) {
		int check = 1;
		for(i=1; i<n && check ;i++) {
			check = (direction*asc_order(&elements[i-1],&elements[i]) <= 0)?1:0;
		}
		
		printf("Check :\t%s\n", (check?"ok":"fail"));

		for(i=0; i<n && i<20;i++) {	printf("%d,",elements[i]); } printf("...\n");
	} 

	/* ����������� ����� ���������� ������� */
	free(elements); 

	MPI_Finalize(); 

	exit(0);
} 


