using System;
using System.Globalization;
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
        public int GridSize
        {
            get { return Convert.ToInt32(numericUpDownGridSize.Value); }
        }

        public int BlockSize
        {
            get { return Convert.ToInt32(numericUpDownBlockSize.Value); }
        }
        private void ValueChanged(object sender, EventArgs e)
        {
            textBox1.Text = (GridSize * BlockSize).ToString(CultureInfo.InvariantCulture);
        }
    }
}