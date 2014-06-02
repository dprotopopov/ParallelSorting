using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;

namespace ParallelSorting.Experiment
{
    public partial class MainForm : Form
    {
        private static readonly Random Rnd = new Random();
        private readonly RunDialog _runDialog = new RunDialog();

        public MainForm()
        {
            InitializeComponent();
            Items = new BindingList<Experiment>();
            dataGridView1.DataSource = Items;
        }

        private BindingList<Experiment> Items { get; set; }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void saveAsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK) return;
            using (var writer = new StreamWriter(File.Open(saveFileDialog1.FileName, FileMode.Create)))
            {
                PropertyInfo[] props = typeof (Experiment).GetProperties();
                writer.WriteLine(string.Join(";", props.Select(p => p.Name)));
                foreach (Experiment item in Items)
                    writer.WriteLine(string.Join(";", props.Select(p => p.GetValue(item, null).ToString())));
            }
        }

        private void runToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (_runDialog.ShowDialog() != DialogResult.OK) return;
            for (int size = _runDialog.FromArraySize;
                size <= _runDialog.ToArraySize;
                size += _runDialog.StepArraySize)
            {
                var experiment = new Experiment();
                string inputFileName = Path.GetTempPath() + Guid.NewGuid() + ".txt";
                string outputFileName = Path.GetTempPath() + Guid.NewGuid() + ".txt";
                int numberMpiProcesses = _runDialog.NumberMpiProcesses;
                int gridSize = _runDialog.GridSize;
                int blockSize = _runDialog.BlockSize;
                using (var writer = new StreamWriter(File.Open(inputFileName, FileMode.Create)))
                    for (int i = 0; i < size; i++)
                        writer.WriteLine(((1 - (Rnd.Next() & 2))*Rnd.Next()).ToString(CultureInfo.InvariantCulture));

                Dictionary<PropertyInfo, string> dictionary = new Dictionary<string, string>
                {
                    {"BitonicMpi", "/C mpiexec -n {0} mpibitonic {3} {4} >> sorting.log"},
                    {"OddevenMpi", "/C mpiexec -n {0} mpioddeven {3} {4} >> sorting.log"},
                    {"BucketMpi", "/C mpiexec -n {0} mpibucket {3} {4} >> sorting.log"},
                    {"BitonicCuda", "/C cudabitonic.exe -g {1} -b {2} {3} {4} >> sorting.log"},
                    {"OddevenCuda", "/C cudaoddeven.exe -g {1} -b {2} {3} {4} >> sorting.log"},
                    {"BucketCuda", "/C cudabucket.exe -g {1} -b {2} {3} {4} >> sorting.log"}
                }
                    .ToDictionary(
                        pair => experiment.GetType().GetProperty(pair.Key),
                        pair =>
                            string.Format(pair.Value, numberMpiProcesses, gridSize, blockSize, inputFileName,
                                outputFileName));
                experiment.ArraySize = size;
                experiment.NumberMpiProcesses = numberMpiProcesses;
                experiment.GridSize = gridSize;
                experiment.BlockSize = blockSize;
                experiment.StartDateTime = DateTime.Now;
                foreach (var pair in dictionary)
                {
                    DateTime start = DateTime.Now;
                    Process process = Process.Start("cmd", pair.Value);
                    process.WaitForExit();
                    DateTime end = DateTime.Now;
                    var timeSpan = new TimeSpan(end.Ticks - start.Ticks);
                    pair.Key.SetValue(experiment, timeSpan, null);
                }
                experiment.EndDateTime = DateTime.Now;
                Items.Add(experiment);
            }
        }
    }
}