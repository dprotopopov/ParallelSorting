using System;

namespace ParallelSorting.Experiment
{
    public class Experiment
    {
        public DateTime StartDateTime { get; set; }
        public DateTime EndDateTime { get; set; }
        public int ArraySize { get; set; }
        public int NumberMpiProcesses { get; set; }
        public int GridSize { get; set; }
        public int BlockSize { get; set; }
        public TimeSpan BitonicMpi { get; set; }
        public TimeSpan OddevenMpi { get; set; }
        public TimeSpan BucketMpi { get; set; }
        public TimeSpan BitonicCuda { get; set; }
        public TimeSpan OddevenCuda { get; set; }
        public TimeSpan BucketCuda { get; set; }
    }
}