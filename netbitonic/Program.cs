using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

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

namespace netbitonic
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
            for (var k = 1; 1 << k <= count; k++)
                if ((count & (1 << k)) != 0)
                    for (var i = 0; i < k; i++)
                    for (var j = i; j >= 0; j--)
                    {
                        var step = 1 << j;
                        var tasks = new List<Task>();

                        for (var loop = 0; loop < numberOfThreads; loop++)
                        {
                            // https://stackoverflow.com/questions/33275831/for-loop-result-in-overflow-with-task-run-or-task-start
                            var t = loop;
                            var task = Task.Run(() =>
                            {
                                Console.WriteLine($"Thread #{t}");
                                for (var id = t; id < 1 << (k - 1); id += numberOfThreads)
                                {
                                    var parity = id >> i;
                                    while (parity > 1) parity = (parity >> 1) ^ (parity & 1);
                                    parity =
                                        1 - (parity <<
                                             1); // теперь переменная parity может иметь только 2 значения 1 и -1
                                    var offset = (count & ((1 << k) - 1)) + ((id >> j) << (j + 1)) +
                                                 (id & ((1 << j) - 1));
                                    if (offset + step < count)
                                        switch (sortOrder * parity)
                                        {
                                            case (int) SortOrder.Asc:
                                                if (list[offset] > list[offset + step])
                                                {
                                                    var x = list[offset];
                                                    list[offset] = list[offset + step];
                                                    list[offset + step] = x;
                                                }
                                                break;
                                            case (int) SortOrder.Desc:
                                                if (list[offset] < list[offset + step])
                                                {
                                                    var x = list[offset];
                                                    list[offset] = list[offset + step];
                                                    list[offset + step] = x;
                                                }
                                                break;
                                        }
                                }
                            });
                            tasks.Add(task);
                        }
                        Task.WaitAll(tasks.ToArray());

                        //Parallel.ForEach(Enumerable.Range(0, numberOfThreads), t =>
                        //{
                        //    for (var id = t; id < 1 << (k - 1); id += numberOfThreads)
                        //    {
                        //        var parity = id >> i;
                        //        while (parity > 1) parity = (parity >> 1) ^ (parity & 1);
                        //        parity =
                        //            1 - (parity << 1); // теперь переменная parity может иметь только 2 значения 1 и -1
                        //        var offset = (count & ((1 << k) - 1)) + ((id >> j) << (j + 1)) + (id & ((1 << j) - 1));
                        //        if (offset + step < count)
                        //            switch (sortOrder * parity)
                        //            {
                        //                case (int) SortOrder.Asc:
                        //                    if (list[offset] > list[offset + step])
                        //                    {
                        //                        var x = list[offset];
                        //                        list[offset] = list[offset + step];
                        //                        list[offset + step] = x;
                        //                    }
                        //                    break;
                        //                case (int) SortOrder.Desc:
                        //                    if (list[offset] < list[offset + step])
                        //                    {
                        //                        var x = list[offset];
                        //                        list[offset] = list[offset + step];
                        //                        list[offset + step] = x;
                        //                    }
                        //                    break;
                        //            }
                        //    }
                        //});
                    }
            // Теперь надо произвести слияние уже отсортированных массивов
            var arr = new long[count];
            var size = new List<int>();
            for (var k = 0; k < 8 * sizeof(int); k++) size.Add(count & (1 << k));

            for (var total = count; total-- > 0;)
            {
                var k = 8 * sizeof(int);
                while (k-- > 0 && size[k] == 0) ;
                for (var i = k; i-- > 0;)
                    if (size[i] > 0)
                        switch (sortOrder)
                        {
                            case (int) SortOrder.Asc:
                                if (list[(count & ((1 << k) - 1)) + size[k] - 1] <
                                    list[(count & ((1 << i) - 1)) + size[i] - 1])
                                    k = i;
                                break;
                            case (int) SortOrder.Desc:
                                if (list[(count & ((1 << k) - 1)) + size[k] - 1] >
                                    list[(count & ((1 << i) - 1)) + size[i] - 1])
                                    k = i;
                                break;
                        }
                size[k]--;
                arr[total] = list[(count & ((1 << k) - 1)) + size[k]];
            }
            list.Clear();
            list.AddRange(arr);
        }

        private static void SortThread(List<long> list, int numberOfThreads, int sortOrder, int t, int k, int i, int j,
            int step, int count)
        {
            Console.WriteLine($"Thread #{t}");
            for (var id = t; id < 1 << (k - 1); id += numberOfThreads)
            {
                var parity = id >> i;
                while (parity > 1) parity = (parity >> 1) ^ (parity & 1);
                parity =
                    1 - (parity << 1); // теперь переменная parity может иметь только 2 значения 1 и -1
                var offset = (count & ((1 << k) - 1)) + ((id >> j) << (j + 1)) + (id & ((1 << j) - 1));
                if (offset + step < count)
                    switch (sortOrder * parity)
                    {
                        case (int) SortOrder.Asc:
                            if (list[offset] > list[offset + step])
                            {
                                var x = list[offset];
                                list[offset] = list[offset + step];
                                list[offset + step] = x;
                            }
                            break;
                        case (int) SortOrder.Desc:
                            if (list[offset] < list[offset + step])
                            {
                                var x = list[offset];
                                list[offset] = list[offset + step];
                                list[offset + step] = x;
                            }
                            break;
                    }
            }
        }
    }
}