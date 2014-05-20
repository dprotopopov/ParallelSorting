using System.Collections.Generic;

namespace ParallelSorting.Editor
{
    class LongComparer : IComparer<long>
    {
        public int Compare(long x, long y)
        {
            if (x < y) return -1;
            if (x > y) return 1;
            return 0;
        }
    }
}