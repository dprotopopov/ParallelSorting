char *title = "bitonic sort";
char *description = "������������ ���������� (bitonic sort)";
/*
� ������ ���� ���������� ����� �������� Bn(��������������, half - cleaner) ��� ��������, �����������
��������������� �������� ��� xi � xi + n / 2.�� ���. 1 �������������� ����� ������������� �������� ��� ��� ��
�����������, ��� � �� ��������.���������� �������� �� ������� ������������ ������������������ �
����������� : ���� ����� ��������������� ��������� ��������� ������������ ������������������ ����� �
������, �� �� ��������� ��������� ������������ ������������������.
������������������ a0, a1, �, an - 1 ���������� ������������, ���� ��� ��� ������� �� ���� ����������
������(�.�.���� ������� ����������, � ����� �������, ���� ��������), ��� �������� ����� ������������
������ �� ����� ������������������.���, ������������������ 5, 7, 6, 4, 2, 1, 3 ������������, ���������
�������� �� 1, 3, 5, 7, 6, 4, 2 ����� ������������ ������ ����� �� ��� ��������.
��������, ��� ���� ��������� �������������� Bn � ������������ ������������������ a0, a1, �, an - 1,
�� ������������ ������������������ �������� ���������� ���������� :
� ��� �� �������� ����� ����� �������������.
� ����� ������� ������ �������� ����� �� ������ ������ �������� ������ ��������.
� ���� �� ���� �� ������� �������� ����������.
�������� � ������������ ������������������ a0, a1, �, an - 1 �������������� Bn, ������� ���
������������������ ������ n / 2, ������ �� ������� ����� ������������, � ������ ������� ������ �� ��������
������ ������� ������.����� �������� � ������ �� ������������ ������� �������������� Bn / 2.�������
��� ������ ������������ ������������������ ����� n / 4.�������� � ������ �� ��� �������������� Bn / 2 �
��������� ���� ������� �� ��� ���, ���� �� ������ � n / 2 ������������������� �� ���� ���������.�������� �
������ �� ��� �������������� B2, ����������� ��� ������������������.��������� ��� ������������������
��� �����������, ��, ��������� ��, ������� ��������������� ������������������.
����, ���������������� ���������� ��������������� Bn, Bn / 2, �, B2 ��������� ������������
������������ ������������������.��� �������� �������� ������������ �������� � ���������� Mn.
��������, � ������������������ �� 8 ��������� a 0, a1, �, a7 �������� �������������� B2, ����� ��
�������� ����� ������� ���������� ��� ��������������.�� ���. 2 �����, ��� ������ ������ ��������
������������ ������������������ �������� ������������ ������������������.���������� ���������
������ �������� ����� �������� ������������ ������������������.������� ������ �� ���� ������� �����
������������� ������������ ��������, ������ �������� ������� ����� �������, ����� �����������
���������� � ��������� ���� ���������������.� ���������� ��� �������� �������� ������ ������������
������������ ���������� ������������������ �� n ��������� ����������� ������� � ������ ��
������� ����������� � ����� �����������.����� ����� ���������� ������������ ������������������
����������� ������������ ��������.
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

/* ������������ ���� ��������� � ������� */
void exchange(const void *e1, const void *e2) 
{ 
	int x = *((int *)e1) ; *((int *)e2) = *((int *)e1); *((int *)e1) = x;
}

/* Bn(��������������, half - cleaner) ��� ��������, �����������
��������������� �������� ��� xi � xi + half */
void Bn (int *elements, int half, int direction) {
	int i;
	for(i = 0; i<half ; i++) {
		if (direction*asc_order(&elements[i],&elements[i+half]) > 0) {
			exchange(&elements[i],&elements[i+half]);
		}
	}
}
/* ���������������� ���������� ��������������� Bn, Bn / 2, �, B2 ��������� ������������
������������ ������������������.��� �������� �������� ������������ �������� � ���������� Mn */
void Mn(int *elements, int half, int direction,int myrank,int nrank) {
	MPI_Status status; 
	Bn(elements, half, direction);
	if (half<2) return;
	if ((myrank+1) < nrank) {
		/* ����������� ������ ��������� ������� */
		int child = 0;
		MPI_Recv(&child, 1, MPI_INT, MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD, &status);
		/* ����� �������� ������� �� ��������� ������� ���������� ��������  */
		MPI_Send(&half, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		MPI_Send(&direction, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD);
		MPI_Send(&elements[half], half, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		Mn(elements,half>>1,direction,myrank,nrank);
		MPI_Recv(&elements[half], half, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD, &status);
	} else {
		/* ������������ �� ���� */
		Mn(elements,half>>1,direction,myrank,nrank);
		Mn(&elements[half>>1],half>>1,direction,myrank,nrank);
	}
}

main(int argc, char *argv[]) 
{ 
	int n;         /* ������ ������������ ������� */ 
	int nrank;      /* ����� ���������� ��������� */ 
	int myrank;    /* ����� �������� �������� */ 
	int *elements;   /* ������ ���������, �������� �������� */ 
	int *elements2;   /* ������ � ������������ ���������� */ 
	int *size;   /* ��������������� ������ */ 
	int direction; /* ������� ���������� 1 - �� �����������, -1 �� �������� */
	int i, j, k; 
	int start, end;
	MPI_Status status; 

	/* ������������ MPI */ 
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

		/* ��������� ������ ������-���������� ������� */ 
		/* �������� ���������� ������ �� ������� �������� */ 
		for (i=0; i<n; i++) {
			elements[i] = random(); 
		}

		// ����� n ����������� � ���� ����� �������� ������,
		// �������, ��������� �������� ������ �� ���������� � ������� ������� ��������� ���� �����
		// � ��������� ������ ��������� ������������ ���������� 
		// � ���������� ������� ������ ����� ���������� ��������������� �������� ������� ������ �������� ������

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

		// ������ ���� ���������� ������� ��� ��������������� ��������
		// ��� ����� �������� ������ ������ �� ������� ��� � ������
		// � ������ �������� ��������
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

		/* ��������� ��� ������� �������� */
		for(i=1; i<nrank;i++) {
			int zero=0;
			int child = 0;
			MPI_Recv(&child, 1, MPI_INT, MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD, &status);
			MPI_Send(&zero, 1, MPI_INT, child, DATA_TAG, MPI_COMM_WORLD); 
		}

	} else {
		while (1==1)
		{
			/* ��� ������� �������� ��������� � ����� �������� ��������� ������� �� �������� �������� */
			/* �������� ������ � �������� ���� ��������� ������������ ������� */
			MPI_Send(&myrank, 1, MPI_INT, MPI_ANY_SOURCE, READY_TAG, MPI_COMM_WORLD); 
			MPI_Recv(&n, 1, MPI_INT, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, &status);
			if (n == 0) {
				/* ������� ������ �� ��������� ������ ��������� */
				break; /* ������� �� ����� �������� */
			} else {
				int parent = status.MPI_SOURCE ;
				MPI_Recv(&direction, 1, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				MPI_Recv(elements, n, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD, &status); 
				Mn(elements, n>>1, direction, myrank, nrank);
				MPI_Send(elements, n, MPI_INT, parent, DATA_TAG, MPI_COMM_WORLD); 
			}
		}
	}


	/* ��������� � ������� ���������� */
	if (myrank == 0) {
		int check = 1;
		for(i=1; i<n && check ;i++) {
			check = (direction*asc_order(&elements2[i-1],&elements2[i]) <= 0)?1:0;
		}

		printf("Check :\t%s\n", (check?"ok":"fail"));

		for(i=0; i<n && i<20;i++) {	printf("%d,",elements2[i]); } printf("...\n");
	} 

	/* ����������� ����� ���������� ������� */
	free(size); 
	free(elements2); 
	free(elements); 

	MPI_Finalize(); 

	exit(0);
} 
