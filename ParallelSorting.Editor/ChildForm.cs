using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace ParallelSorting.Editor
{
    public partial class ChildForm : Form
    {
        private static readonly Random Rnd = new Random();

        public ChildForm()
        {
            InitializeComponent();
        }

        private ChildForm(string s)
        {
            InitializeComponent();
            textBox1.Text = s;
        }

        public static ChildForm OpenFile(string fileName)
        {
            using (var reader = new StreamReader(File.Open(fileName, FileMode.Open)))
                return new ChildForm(reader.ReadToEnd());
        }

        public void SaveAs(string fileName)
        {
            using (var writer = new StreamWriter(File.Open(fileName, FileMode.Create)))
                writer.Write(textBox1.Text);
        }

        public void Random(int count)
        {
            textBox1.Clear();
            for (var i = 0; i < count;)
            {
                var list = new List<string>();
                for (var j = 0; j < 8 && i < count; j++,i++)
                    list.Add(((1 - (Rnd.Next() & 2))*Rnd.Next()).ToString(CultureInfo.InvariantCulture));
                textBox1.AppendText(string.Join("\t", list));
                textBox1.AppendText(Environment.NewLine);
            }
        }

        public void Execute(int numberOfProcess, int gridSize, int blockSize, SortingAlgorithm sortingAlgorithm,
            ExecutionMethod executionMethod)
        {
            var inputFileName = Path.GetTempPath() + Guid.NewGuid() + ".txt";
            var outputFileName = Path.GetTempPath() + Guid.NewGuid() + ".txt";
            var commandFormat = (from object[] item in new object[]
            {
                new object[]
                {
                    SortingAlgorithm.Bitonic, ExecutionMethod.Mpi,
                    "/C mpiexec -n {0} mpibitonic {3} {4} >> sorting.log"
                },
                new object[]
                {
                    SortingAlgorithm.Oddeven, ExecutionMethod.Mpi,
                    "/C mpiexec -n {0} mpioddeven {3} {4} >> sorting.log"
                },
                new object[]
                {
                    SortingAlgorithm.Bucket, ExecutionMethod.Mpi,
                    "/C mpiexec -n {0} mpibucket {3} {4} >> sorting.log"
                },
                new object[]
                {
                    SortingAlgorithm.Bitonic, ExecutionMethod.Cuda,
                    "/C cudabitonic.exe -g {1} -b {2} {3} {4} >> sorting.log"
                },
                new object[]
                {
                    SortingAlgorithm.Oddeven, ExecutionMethod.Cuda,
                    "/C cudaoddeven.exe -g {1} -b {2} {3} {4} >> sorting.log"
                },
                new object[]
                {
                    SortingAlgorithm.Bucket, ExecutionMethod.Cuda,
                    "/C cudabucket.exe -g {1} -b {2} {3} {4} >> sorting.log"
                }
            }
                where (SortingAlgorithm) item[0] == sortingAlgorithm && (ExecutionMethod) item[1] == executionMethod
                select (string) item[2]).FirstOrDefault();
            if (string.IsNullOrEmpty(commandFormat)) throw new NotImplementedException();
            var command = string.Format(commandFormat, numberOfProcess, gridSize, blockSize, inputFileName,
                outputFileName);

            using (var writer = new StreamWriter(File.Open(inputFileName, FileMode.Create)))
                writer.Write(textBox1.Text);

            Debug.WriteLine(command);
            var process = Process.Start("cmd", command);

            if (process == null) return;
            process.WaitForExit();

            if (process.ExitCode != 0) return;
            using (var reader = new StreamReader(File.Open(outputFileName, FileMode.Open)))
                textBox1.Text = reader.ReadToEnd();
        }
    }
}