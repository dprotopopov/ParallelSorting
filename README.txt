��� ���������� MPICH2 ������� � ���������� �� C:
http://www.mpich.org/static/tarballs/1.4.1p1/mpich2-1.4.1p1-win-ia32.msi
http://www.mpich.org/static/tarballs/1.4.1p1/mpich2-1.4.1p1-win-x86-64.msi

��� �������������� CUDA ������� � ���������� �� C:
CUDA 5.5

��� �������������� CUDA ��� ������������� VS ������������ ��������� �������
nvcc -o Release\cudabitonic.exe cudabitonic\butonic.cu
nvcc -o Release\cudaoddeven.exe cudaoddeven\oddeven.cu
nvcc -o Release\cudabucket.exe cudabucket\bucket.cu

��� ������� MPICH2 ��������� ���������:
1. �� ���� ����������� �������� ���������
	smpd.exe -install -phrase ������
2. ������� � ����� Release ���� .smpd � �������
3. �� ��������� ������� � ������, ����������� � ����� Release ������ mpiexec.exe
4. ��������� �����������������, �������� ������� 
	mpiexec.exe -n 10 mpibitonic.exe
	(����� ������������ ��������������� �������� � �.�.)
	����������. MPICH ����������� �����
