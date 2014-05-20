using System;
using System.Windows.Forms;

namespace ParallelSorting.Editor
{
    public partial class RandomDialog : Form
    {
        public RandomDialog()
        {
            InitializeComponent();
        }

        public int Count
        {
            get { return Convert.ToInt32(numericUpDown1.Value); }
        }
    }
}