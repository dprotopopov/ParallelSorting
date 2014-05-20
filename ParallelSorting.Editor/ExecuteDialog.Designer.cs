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
            this.numericUpDown1 = new System.Windows.Forms.NumericUpDown();
            this.radioButtonCuda = new System.Windows.Forms.RadioButton();
            this.radioButtonMpi = new System.Windows.Forms.RadioButton();
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).BeginInit();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.radioButtonBucket);
            this.groupBox1.Controls.Add(this.radioButtonOddeven);
            this.groupBox1.Controls.Add(this.radioButtonBitonic);
            this.groupBox1.Location = new System.Drawing.Point(72, 31);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(348, 140);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Algorithm";
            // 
            // radioButtonBucket
            // 
            this.radioButtonBucket.AutoSize = true;
            this.radioButtonBucket.Location = new System.Drawing.Point(124, 88);
            this.radioButtonBucket.Name = "radioButtonBucket";
            this.radioButtonBucket.Size = new System.Drawing.Size(72, 21);
            this.radioButtonBucket.TabIndex = 2;
            this.radioButtonBucket.TabStop = true;
            this.radioButtonBucket.Text = "Bucket";
            this.radioButtonBucket.UseVisualStyleBackColor = true;
            // 
            // radioButtonOddeven
            // 
            this.radioButtonOddeven.AutoSize = true;
            this.radioButtonOddeven.Location = new System.Drawing.Point(124, 60);
            this.radioButtonOddeven.Name = "radioButtonOddeven";
            this.radioButtonOddeven.Size = new System.Drawing.Size(92, 21);
            this.radioButtonOddeven.TabIndex = 1;
            this.radioButtonOddeven.TabStop = true;
            this.radioButtonOddeven.Text = "Odd-even";
            this.radioButtonOddeven.UseVisualStyleBackColor = true;
            // 
            // radioButtonBitonic
            // 
            this.radioButtonBitonic.AutoSize = true;
            this.radioButtonBitonic.Location = new System.Drawing.Point(124, 32);
            this.radioButtonBitonic.Name = "radioButtonBitonic";
            this.radioButtonBitonic.Size = new System.Drawing.Size(71, 21);
            this.radioButtonBitonic.TabIndex = 0;
            this.radioButtonBitonic.TabStop = true;
            this.radioButtonBitonic.Text = "Bitonic";
            this.radioButtonBitonic.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.numericUpDown1);
            this.groupBox2.Controls.Add(this.radioButtonCuda);
            this.groupBox2.Controls.Add(this.radioButtonMpi);
            this.groupBox2.Location = new System.Drawing.Point(72, 192);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(348, 104);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Platphorm";
            // 
            // numericUpDown1
            // 
            this.numericUpDown1.Location = new System.Drawing.Point(176, 33);
            this.numericUpDown1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numericUpDown1.Name = "numericUpDown1";
            this.numericUpDown1.Size = new System.Drawing.Size(120, 22);
            this.numericUpDown1.TabIndex = 2;
            this.numericUpDown1.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // radioButtonCuda
            // 
            this.radioButtonCuda.AutoSize = true;
            this.radioButtonCuda.Location = new System.Drawing.Point(41, 60);
            this.radioButtonCuda.Name = "radioButtonCuda";
            this.radioButtonCuda.Size = new System.Drawing.Size(67, 21);
            this.radioButtonCuda.TabIndex = 1;
            this.radioButtonCuda.TabStop = true;
            this.radioButtonCuda.Text = "CUDA";
            this.radioButtonCuda.UseVisualStyleBackColor = true;
            // 
            // radioButtonMpi
            // 
            this.radioButtonMpi.AutoSize = true;
            this.radioButtonMpi.Location = new System.Drawing.Point(41, 33);
            this.radioButtonMpi.Name = "radioButtonMpi";
            this.radioButtonMpi.Size = new System.Drawing.Size(52, 21);
            this.radioButtonMpi.TabIndex = 0;
            this.radioButtonMpi.TabStop = true;
            this.radioButtonMpi.Text = "MPI";
            this.radioButtonMpi.UseVisualStyleBackColor = true;
            // 
            // button1
            // 
            this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.button1.Location = new System.Drawing.Point(264, 325);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 2;
            this.button1.Text = "Ok";
            this.button1.UseVisualStyleBackColor = true;
            // 
            // button2
            // 
            this.button2.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.button2.Location = new System.Drawing.Point(345, 325);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(75, 23);
            this.button2.TabIndex = 3;
            this.button2.Text = "Cancel";
            this.button2.UseVisualStyleBackColor = true;
            // 
            // ExecuteDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(521, 380);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Name = "ExecuteDialog";
            this.Text = "ExecuteDialog";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown1)).EndInit();
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
        private System.Windows.Forms.NumericUpDown numericUpDown1;
    }
}