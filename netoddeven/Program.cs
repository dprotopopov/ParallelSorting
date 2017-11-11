using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

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

namespace netoddeven
{
    public enum SortOrder
    {
        Asc = 1,
        Desc = -1
    }

    internal class Program
    {
        private static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage : <program> [-n <numberOfThreads>] [-o <sortOrder>] <inputfile> <outputfile>");
                return;
            }

            var numberOfThreads = Environment.ProcessorCount;
            var sortOrder = (int) SortOrder.Asc;
            string inputFileName;
            string outputFileName;

            try
            {
                var argId = 0;
                for (; argId < args.Length && args[argId][0] == '-'; argId++)
                    switch (args[argId][1])
                    {
                        case 'n':
                            int.TryParse(args[++argId], out numberOfThreads);
                            break;
                        case 'o':
                            int.TryParse(args[++argId], out sortOrder);
                            break;
                    }
                // Получаем параметры - имена файлов
                inputFileName = args[argId++];
                outputFileName = args[argId++];
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return;
            }

            var list = new List<long>();
            var spaces = new Regex(@"\s+");
            using (var reader = new StreamReader(File.Open(inputFileName, FileMode.Open)))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var fields = spaces.Split(line);
                    foreach (var digits in fields)
                        if (!string.IsNullOrEmpty(digits))
                        {
                            long.TryParse(digits, out var value);
                            list.Add(value);
                        }
                }
            }

            Sort(list, numberOfThreads, sortOrder);

            using (var writer = new StreamWriter(File.Open(outputFileName, FileMode.Create)))
            {
                foreach (var value in list)
                    writer.WriteLine(value);
            }
        }

        private static void Sort(List<long> list, int numberOfThreads, int sortOrder)
        {
            var count = list.Count;
            var blockSize = (list.Count + 2 * numberOfThreads) / (2 * numberOfThreads + 1);
            var numberOfIterations = numberOfThreads + 2;
            for (var j = 0; j < numberOfIterations; j++)
            {
                var parity = j & 1;
                Parallel.ForEach(Enumerable.Range(0, numberOfThreads), i =>
                {
                    var index = blockSize * (2 * i + parity);
                    var twoBlocks = new List<long>();
                    for (var k = index; k < index + 2 * blockSize && k < count; k++)
                        twoBlocks.Add(list[k]);

                    switch (sortOrder)
                    {
                        case (int) SortOrder.Asc:
                            twoBlocks = new List<long>(twoBlocks.OrderBy(x => x));
                            break;
                        case (int) SortOrder.Desc:
                            twoBlocks = new List<long>(twoBlocks.OrderByDescending(x => x));
                            break;
                    }

                    for (var k = index; k < index + 2 * blockSize && k < count; k++)
                        list[k] = twoBlocks[k - index];
                });
            }
        }
    }
}