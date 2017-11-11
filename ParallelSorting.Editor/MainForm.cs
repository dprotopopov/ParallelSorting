using System;
using System.Windows.Forms;

namespace ParallelSorting.Editor
{
    public partial class MainForm : Form
    {
        private static readonly ExecuteDialog ExecuteDialog = new ExecuteDialog();

        private static readonly RandomDialog RandomDialog = new RandomDialog();

        public MainForm()
        {
            InitializeComponent();
        }

        private void newToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = new ChildForm
            {
                MdiParent = this
            };
            child.Show();
        }

        private void randomToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as ChildForm;
            if (child == null) return;
            if (RandomDialog.ShowDialog() != DialogResult.OK) return;
            child.Random(RandomDialog.Count);
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() != DialogResult.OK) return;
            var child = ChildForm.OpenFile(openFileDialog1.FileName);
            child.MdiParent = this;
            child.Show();
        }

        private void saveAsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as ChildForm;
            if (child == null) return;
            if (saveFileDialog1.ShowDialog() != DialogResult.OK) return;
            child.SaveAs(saveFileDialog1.FileName);
        }

        private void executeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as ChildForm;
            if (child == null) return;
            if (ExecuteDialog.ShowDialog() != DialogResult.OK) return;
            if (!ExecuteDialog.IsValid()) return;
            child.Execute(ExecuteDialog.NumberOfProcess, ExecuteDialog.GridSize, ExecuteDialog.BlockSize,
                ExecuteDialog.DegreeOfParallelism,
                ExecuteDialog.SortingAlgorithm, ExecuteDialog.ExecutionMethod);
        }
    }
}