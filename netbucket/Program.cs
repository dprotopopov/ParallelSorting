using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

/*
В блочной карманной или корзинной сортировке(Bucket sort) сортируемые элементы распределены
между конечным числом отдельных блоков(карманов, корзин).Каждый блок затем сортируется отдельно либо
рекурсивно тем же методом либо другим. Затем элементы помещают обратно в массив.
Для этой сортировки характерно линейное время исполнения.
Алгоритм требует знаний о природе сортируемых данных, выходящих за рамки функций "сравнить" и
"поменять местами", достаточных для сортировки слиянием, сортировки пирамидой, быстрой сортировки,
сортировки Шелла, сортировки вставкой.
*/

namespace netbucket
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
                Console.WriteLine(
                    "Usage : <program> [-n <numberOfThreads>] [-o <sortOrder>] [-b <numberOfBuckets>] <inputfile> <outputfile>");
                return;
            }

            var numberOfThreads = Environment.ProcessorCount;
            var numberOfBuckets = 257;
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
                        case 'b':
                            int.TryParse(args[++argId], out numberOfBuckets);
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

            Sort(list, numberOfBuckets, numberOfThreads, sortOrder);

            using (var writer = new StreamWriter(File.Open(outputFileName, FileMode.Create)))
            {
                foreach (var value in list)
                    writer.WriteLine(value);
            }
        }

        private static void Sort(List<long> list, int numberOfBuckets, int numberOfThreads, int sortOrder,
            long low = long.MinValue, long high = long.MaxValue)
        {
            if (low >= high) return;
            var count = list.Count();
            var buckets = new List<List<long>>();
            for (var i = 0; i < numberOfBuckets; i++) buckets.Add(new List<long>());
            Parallel.ForEach(Enumerable.Range(0, numberOfThreads), t =>
            {
                for (var index = t; index < count; index += numberOfThreads)
                {
                    var value = list[index];

                    // Определяем номер корзины
                    var bucketIndex = numberOfBuckets * ((decimal) value - low) / ((decimal) high + 1 - low);

                    // Добавляем элемент в корзину
                    buckets[(int) bucketIndex].Add(value);
                }
            });

            list.Clear();

            for (var index = 0; index < numberOfBuckets; index++)
            {
                if (buckets.ElementAt(index).Count < 2) continue;
                var bucketLow = low + index * ((decimal) high + 1 - low) / numberOfBuckets;
                var bucketHigh = low + (index + 1) * ((decimal) high + 1 - low) / numberOfBuckets - 1;
                Sort(buckets.ElementAt(index), numberOfBuckets, numberOfThreads, sortOrder, (long) bucketLow,
                    (long) bucketHigh);
            }

            switch (sortOrder)
            {
                case (int) SortOrder.Asc:
                    for (var i = 0; i < numberOfBuckets; i++)
                        list.AddRange(buckets.ElementAt(i));
                    break;
                case (int) SortOrder.Desc:
                    for (var i = numberOfBuckets; i-- > 0;)
                        list.AddRange(buckets.ElementAt(i));
                    break;
            }
        }
    }
}