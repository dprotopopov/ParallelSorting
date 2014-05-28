Для компиляции MPICH2 скачать и установить на C:
http://www.mpich.org/static/tarballs/1.4.1p1/mpich2-1.4.1p1-win-ia32.msi
http://www.mpich.org/static/tarballs/1.4.1p1/mpich2-1.4.1p1-win-x86-64.msi

Для компилирования CUDA скачать и установить на C:
CUDA 5.5

Для компилирования CUDA без использования VS использовать следующие команды
nvcc -o Release\cudabitonic.exe cudabitonic\butonic.cu
nvcc -o Release\cudaoddeven.exe cudaoddeven\oddeven.cu
nvcc -o Release\cudabucket.exe cudabucket\bucket.cu

Для запуска MPICH2 выполнить следующее:
1. На всех компьютерах кластера выполнить
	smpd.exe -install -phrase пароль
2. Создать в папке Release файл .smpd с паролем
3. Во избежание путаниц с путями, скопировать в папку Release нужный mpiexec.exe
4. Проверить работоспособность, выполнив команду 
	mpiexec.exe -n 10 mpibitonic.exe
	(может потребоватся разблокирование фаревола и т.д.)
	Примечание. MPICH запускается долго
