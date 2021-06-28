using System;
using OpenCvSharp;

namespace GestureRecognition
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            int i = 0;
            var video = new VideoCapture();
            video.Open(0);
            if (!video.IsOpened())
            {
                Console.WriteLine("camera open failed.");
                return;
            }

            Console.WriteLine("camera open success.");
            var camera = new Mat();
            while (true)
            {
                i++;
                video.Read(camera);
                if (camera.Empty()) break;
                //Cv2.CvtColor(camera, grayMat, ColorConversionCodes.RGB2BGR);
                //Cv2.Canny(grayMat, camera, 100, 200);
                var skin = SkinDetect(camera);
                Cv2.CvtColor(skin,skin,ColorConversionCodes.GRAY2BGR);
                Cv2.BitwiseAnd(camera,skin,camera);
                var cameraWindow = new Window("cameraSteam", camera, WindowFlags.AutoSize);
                var backgroundWindow = new Window("background", skin, WindowFlags.AutoSize);
                Cv2.WaitKey(10);
                if (i >= 20)
                {
                    i = 0;
                    GC.Collect();
                }
            }
            camera.Release();
            Cv2.DestroyAllWindows();
        }

        public static Mat SkinDetect(Mat input)
        {
            var output = new Mat();
            Cv2.CvtColor(input,output,ColorConversionCodes.BGR2YCrCb);
            Cv2.Split(output, out Mat[] frames);
            var crFrame = frames[1];
            Cv2.GaussianBlur(crFrame,crFrame,new Size(5,5),0);
            Cv2.Threshold(crFrame, crFrame, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);
            return crFrame;
        }
    }
}