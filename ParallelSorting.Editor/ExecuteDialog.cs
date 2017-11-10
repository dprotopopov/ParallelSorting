using System;
using System.Globalization;
using System.Windows.Forms;

namespace ParallelSorting.Editor
{
    public partial class ExecuteDialog : Form
    {
        public ExecuteDialog()
        {
            InitializeComponent();
        }

        public SortingAlgorithm SortingAlgorithm
        {
            get
            {
                if (IsBitonic) return SortingAlgorithm.Bitonic;
                if (IsOddeven) return SortingAlgorithm.Oddeven;
                if (IsBucket) return SortingAlgorithm.Bucket;
                throw new NotImplementedException();
            }
        }

        public ExecutionMethod ExecutionMethod
        {
            get
            {
                if (IsMpi) return ExecutionMethod.Mpi;
                if (IsCuda) return ExecutionMethod.Cuda;
                throw new NotImplementedException();
            }
        }

        private bool IsBitonic
        {
            get { return radioButtonBitonic.Checked; }
        }

        private bool IsOddeven
        {
            get { return radioButtonOddeven.Checked; }
        }

        private bool IsBucket
        {
            get { return radioButtonBucket.Checked; }
        }

        private bool IsCuda
        {
            get { return radioButtonCuda.Checked; }
        }

        private bool IsMpi
        {
            get { return radioButtonMpi.Checked; }
        }

        public int NumberOfProcess
        {
            get { return Convert.ToInt32(numericUpDownNumberOfProcess.Value); }
            set { numericUpDownNumberOfProcess.Value = value; }
        }

        public int GridSize
        {
            get { return Convert.ToInt32(numericUpDownGridSize.Value); }
            set { numericUpDownGridSize.Value = value; }
        }

        public int BlockSize
        {
            get { return Convert.ToInt32(numericUpDownBlockSize.Value); }
            set { numericUpDownBlockSize.Value = value; }
        }


        public bool IsValid()
        {
            return (IsCuda ^ IsMpi) &&
                   (IsBitonic ^ IsOddeven ^ IsBucket);
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            textBox1.Text = (GridSize*BlockSize).ToString(CultureInfo.InvariantCulture);
        }
    }
}