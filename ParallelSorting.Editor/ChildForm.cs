using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using MyLibrary.Collections;

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
            for (int i = 0; i < count;)
            {
                var list = new StackListQueue<string>();
                for (int j = 0; j < 8 && i < count; j++,i++)
                    list.Add(((1 - (Rnd.Next() & 2))*Rnd.Next()).ToString(CultureInfo.InvariantCulture));
                textBox1.AppendText(string.Join("\t", list));
                textBox1.AppendText(Environment.NewLine);
            }
        }

        public void Execute(int numberOfProcess, SortingAlgorithm sortingAlgorithm, ExecutionMethod executionMethod)
        {
            string inputFileName = Path.GetTempPath() + Guid.NewGuid() + ".txt";
            string outputFileName = Path.GetTempPath() + Guid.NewGuid() + ".txt";
            string commandFormat = (from object[] item in new object[]
            {
                new object[]
                {
                    SortingAlgorithm.Bitonic, ExecutionMethod.Mpi,
                    "/C mpiexec.exe -n {0} mpibitonic {1} {2} >> sorting.log"
                },
                new object[]
                {
                    SortingAlgorithm.Oddeven, ExecutionMethod.Mpi,
                    "/C mpiexec.exe -n {0} mpioddeven {1} {2} >> sorting.log"
                },
                new object[]
                {SortingAlgorithm.Bucket, ExecutionMethod.Mpi, "/C mpiexec.exe -n {0} mpibucket {1} {2} >> sorting.log"},
                new object[]
                {SortingAlgorithm.Bitonic, ExecutionMethod.Cuda, "/C cudabitonic.exe {1} {2} >> sorting.log"},
                new object[]
                {SortingAlgorithm.Oddeven, ExecutionMethod.Cuda, "/C cudaoddeven.exe {1} {2} >> sorting.log"},
                new object[]
                {SortingAlgorithm.Bucket, ExecutionMethod.Cuda, "/C cudabucket.exe {1} {2} >> sorting.log"}
            }
                where (SortingAlgorithm) item[0] == sortingAlgorithm && (ExecutionMethod) item[1] == executionMethod
                select (string) item[2]).FirstOrDefault();
            if (string.IsNullOrEmpty(commandFormat)) throw new NotImplementedException();
            string command = string.Format(commandFormat, numberOfProcess, inputFileName, outputFileName);

            using (var writer = new StreamWriter(File.Open(inputFileName, FileMode.Create)))
                writer.Write(textBox1.Text);

            Debug.WriteLine(command);
            Process process = Process.Start("cmd", command);

            if (process == null) return;
            process.WaitForExit();

            if (process.ExitCode != 0) return;
            using (var reader = new StreamReader(File.Open(outputFileName, FileMode.Open)))
                textBox1.Text = reader.ReadToEnd();
        }

        public void Check()
        {
            var list =
                new SortedStackListQueue<long>(
                    (from Match match in Regex.Matches(textBox1.Text, @"[-]?\d+") select Convert.ToInt32(match.Value))
                        .Select(dummy => (long) dummy))
                {
                    Comparer = new LongComparer()
                };
            MessageBox.Show(list.IsSorted(list) ? "Sorted" : "No sorted");
        }
    }
}