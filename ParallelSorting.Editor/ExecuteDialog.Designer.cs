namespace ParallelSorting.Editor
{
    partial class ExecuteDialog
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.radioButtonBucket = new System.Windows.Forms.RadioButton();
            this.radioButtonOddeven = new System.Windows.Forms.RadioButton();
            this.radioButtonBitonic = new System.Windows.Forms.RadioButton();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.numericUpDownBlockSize = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownGridSize = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownNumberOfProcess = new System.Windows.Forms.NumericUpDown();
            this.radioButtonCuda = new System.Windows.Forms.RadioButton();
            this.radioButtonMpi = new System.Windows.Forms.RadioButton();
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.numericUpDownDegreeOfParallelism = new System.Windows.Forms.NumericUpDown();
            this.radioButtonNet = new System.Windows.Forms.RadioButton();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownBlockSize)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownGridSize)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumberOfProcess)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownDegreeOfParallelism)).BeginInit();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.radioButtonBucket);
            this.groupBox1.Controls.Add(this.radioButtonOddeven);
            this.groupBox1.Controls.Add(this.radioButtonBitonic);
            this.groupBox1.Location = new System.Drawing.Point(81, 39);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox1.Size = new System.Drawing.Size(586, 175);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Алгоритм";
            // 
            // radioButtonBucket
            // 
            this.radioButtonBucket.AutoSize = true;
            this.radioButtonBucket.Location = new System.Drawing.Point(176, 106);
            this.radioButtonBucket.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.radioButtonBucket.Name = "radioButtonBucket";
            this.radioButtonBucket.Size = new System.Drawing.Size(207, 24);
            this.radioButtonBucket.TabIndex = 2;
            this.radioButtonBucket.TabStop = true;
            this.radioButtonBucket.Text = "Корзинная сортировка";
            this.radioButtonBucket.UseVisualStyleBackColor = true;
            // 
            // radioButtonOddeven
            // 
            this.radioButtonOddeven.AutoSize = true;
            this.radioButtonOddeven.Location = new System.Drawing.Point(176, 71);
            this.radioButtonOddeven.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.radioButtonOddeven.Name = "radioButtonOddeven";
            this.radioButtonOddeven.Size = new System.Drawing.Size(251, 24);
            this.radioButtonOddeven.TabIndex = 1;
            this.radioButtonOddeven.TabStop = true;
            this.radioButtonOddeven.Text = "Чётно-нечётная сортировка";
            this.radioButtonOddeven.UseVisualStyleBackColor = true;
            // 
            // radioButtonBitonic
            // 
            this.radioButtonBitonic.AutoSize = true;
            this.radioButtonBitonic.Location = new System.Drawing.Point(176, 36);
            this.radioButtonBitonic.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.radioButtonBitonic.Name = "radioButtonBitonic";
            this.radioButtonBitonic.Size = new System.Drawing.Size(234, 24);
            this.radioButtonBitonic.TabIndex = 0;
            this.radioButtonBitonic.TabStop = true;
            this.radioButtonBitonic.Text = "Битоническая сортировка";
            this.radioButtonBitonic.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.numericUpDownDegreeOfParallelism);
            this.groupBox2.Controls.Add(this.radioButtonNet);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.textBox1);
            this.groupBox2.Controls.Add(this.numericUpDownBlockSize);
            this.groupBox2.Controls.Add(this.numericUpDownGridSize);
            this.groupBox2.Controls.Add(this.numericUpDownNumberOfProcess);
            this.groupBox2.Controls.Add(this.radioButtonCuda);
            this.groupBox2.Controls.Add(this.radioButtonMpi);
            this.groupBox2.Location = new System.Drawing.Point(81, 240);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.groupBox2.Size = new System.Drawing.Size(586, 188);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Платформа";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(317, 77);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(18, 20);
            this.label2.TabIndex = 7;
            this.label2.Text = "=";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(237, 79);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(16, 20);
            this.label1.TabIndex = 6;
            this.label1.Text = "x";
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(342, 74);
            this.textBox1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textBox1.Name = "textBox1";
            this.textBox1.ReadOnly = true;
            this.textBox1.Size = new System.Drawing.Size(68, 26);
            this.textBox1.TabIndex = 5;
            // 
            // numericUpDownBlockSize
            // 
            this.numericUpDownBlockSize.Location = new System.Drawing.Point(258, 74);
            this.numericUpDownBlockSize.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownBlockSize.Maximum = new decimal(new int[] {
            1024,
            0,
            0,
            0});
            this.numericUpDownBlockSize.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownBlockSize.Name = "numericUpDownBlockSize";
            this.numericUpDownBlockSize.Size = new System.Drawing.Size(53, 26);
            this.numericUpDownBlockSize.TabIndex = 4;
            this.numericUpDownBlockSize.Value = new decimal(new int[] {
            15,
            0,
            0,
            0});
            // 
            // numericUpDownGridSize
            // 
            this.numericUpDownGridSize.Location = new System.Drawing.Point(176, 74);
            this.numericUpDownGridSize.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownGridSize.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numericUpDownGridSize.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownGridSize.Name = "numericUpDownGridSize";
            this.numericUpDownGridSize.Size = new System.Drawing.Size(54, 26);
            this.numericUpDownGridSize.TabIndex = 3;
            this.numericUpDownGridSize.Value = new decimal(new int[] {
            15,
            0,
            0,
            0});
            this.numericUpDownGridSize.Validated += new System.EventHandler(this.ValueChanged);
            // 
            // numericUpDownNumberOfProcess
            // 
            this.numericUpDownNumberOfProcess.Location = new System.Drawing.Point(176, 40);
            this.numericUpDownNumberOfProcess.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownNumberOfProcess.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownNumberOfProcess.Name = "numericUpDownNumberOfProcess";
            this.numericUpDownNumberOfProcess.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownNumberOfProcess.TabIndex = 2;
            this.numericUpDownNumberOfProcess.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // radioButtonCuda
            // 
            this.radioButtonCuda.AutoSize = true;
            this.radioButtonCuda.Location = new System.Drawing.Point(46, 74);
            this.radioButtonCuda.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.radioButtonCuda.Name = "radioButtonCuda";
            this.radioButtonCuda.Size = new System.Drawing.Size(80, 24);
            this.radioButtonCuda.TabIndex = 1;
            this.radioButtonCuda.TabStop = true;
            this.radioButtonCuda.Text = "CUDA";
            this.radioButtonCuda.UseVisualStyleBackColor = true;
            // 
            // radioButtonMpi
            // 
            this.radioButtonMpi.AutoSize = true;
            this.radioButtonMpi.Location = new System.Drawing.Point(46, 41);
            this.radioButtonMpi.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.radioButtonMpi.Name = "radioButtonMpi";
            this.radioButtonMpi.Size = new System.Drawing.Size(62, 24);
            this.radioButtonMpi.TabIndex = 0;
            this.radioButtonMpi.TabStop = true;
            this.radioButtonMpi.Text = "MPI";
            this.radioButtonMpi.UseVisualStyleBackColor = true;
            // 
            // button1
            // 
            this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.button1.Location = new System.Drawing.Point(494, 436);
            this.button1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(84, 29);
            this.button1.TabIndex = 2;
            this.button1.Text = "Ok";
            this.button1.UseVisualStyleBackColor = true;
            // 
            // button2
            // 
            this.button2.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.button2.Location = new System.Drawing.Point(585, 436);
            this.button2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(84, 29);
            this.button2.TabIndex = 3;
            this.button2.Text = "Cancel";
            this.button2.UseVisualStyleBackColor = true;
            // 
            // numericUpDownDegreeOfParallelism
            // 
            this.numericUpDownDegreeOfParallelism.Location = new System.Drawing.Point(176, 108);
            this.numericUpDownDegreeOfParallelism.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.numericUpDownDegreeOfParallelism.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDownDegreeOfParallelism.Name = "numericUpDownDegreeOfParallelism";
            this.numericUpDownDegreeOfParallelism.Size = new System.Drawing.Size(135, 26);
            this.numericUpDownDegreeOfParallelism.TabIndex = 9;
            this.numericUpDownDegreeOfParallelism.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // radioButtonNet
            // 
            this.radioButtonNet.AutoSize = true;
            this.radioButtonNet.Location = new System.Drawing.Point(46, 109);
            this.radioButtonNet.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.radioButtonNet.Name = "radioButtonNet";
            this.radioButtonNet.Size = new System.Drawing.Size(59, 24);
            this.radioButtonNet.TabIndex = 8;
            this.radioButtonNet.TabStop = true;
            this.radioButtonNet.Text = "Net";
            this.radioButtonNet.UseVisualStyleBackColor = true;
            // 
            // ExecuteDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(760, 494);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Name = "ExecuteDialog";
            this.Text = "ExecuteDialog";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownBlockSize)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownGridSize)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumberOfProcess)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownDegreeOfParallelism)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.RadioButton radioButtonBucket;
        private System.Windows.Forms.RadioButton radioButtonOddeven;
        private System.Windows.Forms.RadioButton radioButtonBitonic;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.RadioButton radioButtonCuda;
        private System.Windows.Forms.RadioButton radioButtonMpi;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.NumericUpDown numericUpDownNumberOfProcess;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.NumericUpDown numericUpDownBlockSize;
        private System.Windows.Forms.NumericUpDown numericUpDownGridSize;
        private System.Windows.Forms.NumericUpDown numericUpDownDegreeOfParallelism;
        private System.Windows.Forms.RadioButton radioButtonNet;
    }
}