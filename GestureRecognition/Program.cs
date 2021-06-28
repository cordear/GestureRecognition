using System;
using OpenCvSharp;

namespace GestureRecognition
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var i = 0;
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
                var contours = FindContours(skin);
                Cv2.CvtColor(skin, skin, ColorConversionCodes.GRAY2BGR);
                Cv2.BitwiseAnd(camera, skin, camera);
                Cv2.DrawContours(camera, contours, -1, Scalar.Red);
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
            Cv2.CvtColor(input, output, ColorConversionCodes.BGR2YCrCb);
            Cv2.Split(output, out var frames);
            var crFrame = frames[1];
            Cv2.GaussianBlur(crFrame, crFrame, new Size(5, 5), 0);
            Cv2.Threshold(crFrame, crFrame, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);
            return crFrame;
        }

        public static Point[][] FindContours(Mat input)
        {
            Cv2.FindContours(input, out var contours, out var hierarchyIndices, RetrievalModes.Tree,
                ContourApproximationModes.ApproxSimple);
            return contours;
        }
    }
}