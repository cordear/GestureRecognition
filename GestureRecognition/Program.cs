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
            var background = new Mat();
            var es = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
            var mog = BackgroundSubtractorMOG2.Create();
            while (true)
            {
                video.Read(camera);
                if (camera.Empty()) break;
                //Cv2.CvtColor(camera, grayMat, ColorConversionCodes.RGB2BGR);
                //Cv2.Canny(grayMat, camera, 100, 200);
                mog.Apply(camera,background);
                Cv2.Threshold(background, background, 244, 255, ThresholdTypes.Binary);
                Cv2.Dilate(background,background,es,iterations:2);
                Cv2.CvtColor(background,background,ColorConversionCodes.GRAY2BGR);
                Cv2.BitwiseAnd(camera,background,camera);
                var cameraWindow = new Window("cameraSteam", camera, WindowFlags.AutoSize);
                var backgroundWindow = new Window("background", background, WindowFlags.AutoSize);
                Cv2.WaitKey(10);
            }
            camera.Release();
            Cv2.DestroyAllWindows();
        }
    }
}