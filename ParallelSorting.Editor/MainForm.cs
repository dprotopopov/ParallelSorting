using System;
using System.Windows.Forms;

namespace ParallelSorting.Editor
{
    public partial class MainForm : Form
    {
        private readonly ExecuteDialog _executeDialog = new ExecuteDialog();
        private readonly RandomDialog _randomDialog = new RandomDialog();

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
            if (_randomDialog.ShowDialog() != DialogResult.OK) return;
            child.Random(_randomDialog.Count);
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() != DialogResult.OK) return;
            ChildForm child = ChildForm.OpenFile(openFileDialog1.FileName);
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
            if (_executeDialog.ShowDialog() != DialogResult.OK) return;
            if (!_executeDialog.IsValid()) return;
            child.Execute(_executeDialog.NumberOfProcess, _executeDialog.GridSize, _executeDialog.BlockSize,
                _executeDialog.SortingAlgorithm, _executeDialog.ExecutionMethod);
        }

        private void checkToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var child = ActiveMdiChild as ChildForm;
            if (child == null) return;
            child.Check();
        }
    }
}