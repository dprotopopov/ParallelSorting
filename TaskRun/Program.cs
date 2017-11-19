using System;
using System.Threading.Tasks;

namespace TaskRun
{
    internal class Program
    {
        public static void Main()
        {
            Console.WriteLine("Starting.");

            for (var i = 0; i < 4; ++i)
            {
                var j = i;
                Task.Run(() => Console.WriteLine(j));
            }

            Console.WriteLine("Finished. Press <ENTER> to exit.");
            Console.ReadLine();
        }
    }
}