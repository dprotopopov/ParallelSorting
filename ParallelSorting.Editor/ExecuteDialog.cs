﻿using System;
using System.Windows.Forms;
using Boolean = MyLibrary.Types.Boolean;

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
            get { return Convert.ToInt32(numericUpDown1.Value); }
        }

        public bool IsValid()
        {
            return Boolean.Xor(IsCuda, IsMpi) &&
                   Boolean.Xor(IsBitonic, IsOddeven, IsBucket);
        }
    }
}