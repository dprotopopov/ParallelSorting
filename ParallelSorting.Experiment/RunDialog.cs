using System;
using System.Windows.Forms;

namespace ParallelSorting.Experiment
{
    public partial class RunDialog : Form
    {
        public RunDialog()
        {
            InitializeComponent();
        }

        public int FromArraySize
        {
            get { return Convert.ToInt32(numericUpDown1.Value); }
        }

        public int ToArraySize
        {
            get { return Convert.ToInt32(numericUpDown2.Value); }
        }

        public int StepArraySize
        {
            get { return Convert.ToInt32(numericUpDown3.Value); }
        }

        public int NumberMpiProcesses
        {
            get { return Convert.ToInt32(numericUpDown4.Value); }
        }
    }
}