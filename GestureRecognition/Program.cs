using System;
using System.Xml;
using OpenCvSharp;

namespace GestureRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var video = new VideoCapture();
            video.Open(0);
            if (!video.IsOpened())
            {
                Console.WriteLine("camera open failed.");
                return;
            }
            else
            {
                Console.WriteLine("camera open success.");
            }
            Mat camera = new Mat();
            while (true)
            {
                video.Read(camera);
                if (camera.Empty())
                {
                    break;
                }

                var cameraWindow = new Window("cameraSteam",camera,WindowFlags.AutoSize);
                Cv2.WaitKey(10);
            }
        }
    }
}
