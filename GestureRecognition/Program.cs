using System;
using OpenCvSharp;

namespace GestureRecognition
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var video = new VideoCapture();
            video.Open(0);
            if (!video.IsOpened())
            {
                Console.WriteLine("camera open failed.");
                return;
            }

            Console.WriteLine("camera open success.");
            var camera = new Mat();
            var grayMat = new Mat();
            while (true)
            {
                video.Read(camera);
                if (camera.Empty()) break;
                Cv2.CvtColor(camera, grayMat, ColorConversionCodes.RGB2BGR);
                Cv2.Canny(grayMat, camera, 100, 200);
                var cameraWindow = new Window("cameraSteam", camera, WindowFlags.AutoSize);
                Cv2.WaitKey(10);
            }
        }
    }
}