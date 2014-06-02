using System;
using System.Globalization;
using System.Windows.Forms;
using MiniMax.Forms;
using Boolean = MyLibrary.Types.Boolean;

namespace ParallelSorting.Editor
{
    public partial class ExecuteDialog : Form
    {
        public ExecuteDialog()
        {
            InitializeComponent();
        }

        public BuildChooseDialog CudaBuildChooseDialog { get; set; }

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
            return Boolean.Xor(IsCuda, IsMpi) &&
                   Boolean.Xor(IsBitonic, IsOddeven, IsBucket);
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            textBox1.Text = (GridSize*BlockSize).ToString(CultureInfo.InvariantCulture);
        }

        private void buttonCudaChoose_Click(object sender, EventArgs e)
        {
            if (CudaBuildChooseDialog.ShowDialog() != DialogResult.OK) return;
            MyLibrary.Collections.Properties values = CudaBuildChooseDialog.Values;
            GridSize = 1;
            BlockSize = Convert.ToInt32(values["N"]);
        }
    }
}