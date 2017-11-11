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
                if (IsNet) return ExecutionMethod.Net;
                throw new NotImplementedException();
            }
        }

        private bool IsBitonic => radioButtonBitonic.Checked;

        private bool IsOddeven => radioButtonOddeven.Checked;

        private bool IsBucket => radioButtonBucket.Checked;

        private bool IsCuda => radioButtonCuda.Checked;

        private bool IsMpi => radioButtonMpi.Checked;

        private bool IsNet => radioButtonNet.Checked;

        public int NumberOfProcess
        {
            get => Convert.ToInt32(numericUpDownNumberOfProcess.Value);
            set => numericUpDownNumberOfProcess.Value = value;
        }

        public int DegreeOfParallelism
        {
            get => Convert.ToInt32(numericUpDownDegreeOfParallelism.Value);
            set => numericUpDownDegreeOfParallelism.Value = value;
        }

        public int GridSize
        {
            get => Convert.ToInt32(numericUpDownGridSize.Value);
            set => numericUpDownGridSize.Value = value;
        }

        public int BlockSize
        {
            get => Convert.ToInt32(numericUpDownBlockSize.Value);
            set => numericUpDownBlockSize.Value = value;
        }


        public bool IsValid()
        {
            return IsCuda ^ IsMpi ^ IsNet &&
                   IsBitonic ^ IsOddeven ^ IsBucket;
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            textBox1.Text = (GridSize * BlockSize).ToString(CultureInfo.InvariantCulture);
        }
    }
}