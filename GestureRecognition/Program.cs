using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using OpenCvSharp;

namespace GestureRecognition
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var i = 0;
            var acuteAngle = 0;
            var video = new VideoCapture();
            video.Open(0);
            if (!video.IsOpened())
            {
                Console.WriteLine("camera open failed.");
                return;
            }

            Console.WriteLine("camera open success.");
            var camera = new Mat();
            var element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
            var hull = new List<Point[]>();
            var hullI = new List<int>();
            while (true)
            {
                i++;
                video.Read(camera);
                if (camera.Empty()) break;
                //Cv2.CvtColor(camera, grayMat, ColorConversionCodes.RGB2BGR);
                //Cv2.Canny(grayMat, camera, 100, 200);
                Cv2.Flip(camera, camera, FlipMode.Y);
                var originWindow = new Window("Origin", camera, WindowFlags.AutoSize);
                var skin = SkinDetect(camera);
                // skin layer process
                //Cv2.Erode(skin, skin, element, iterations: 2);
                Cv2.MorphologyEx(skin, skin, MorphTypes.Open, element,iterations:6);
                //Cv2.Dilate(skin, skin, element, iterations: 2);
                Cv2.MorphologyEx(skin, skin, MorphTypes.Close, element,iterations:6);
                //Cv2.Dilate(skin, skin, element, iterations: 2);
                Cv2.MorphologyEx(skin, skin, MorphTypes.Close, element,iterations:6);
                //Cv2.Erode(skin, skin, element, iterations: 2);
                Cv2.MorphologyEx(skin, skin, MorphTypes.Open, element,iterations:6);

                var contours = FindContours(skin);

                Cv2.CvtColor(skin, skin, ColorConversionCodes.GRAY2BGR);
                Cv2.BitwiseAnd(camera, skin, camera);

                foreach (var t in contours)
                {
                    hull.Add(Cv2.ConvexHull(t));
                    OutputArray outArr = OutputArray.Create(hullI);
                    Cv2.ConvexHull(InputArray.Create(t), outArr, true, false);
                    var defects = Cv2.ConvexityDefects(t, hullI);
                    for (var k = 0; k < defects.Length; k++)
                    {
                        var start = t[defects[k].Item0];
                        var end = t[defects[k].Item1];
                        var far = t[defects[k].Item2];

                        var depth = defects[k].Item3 / 256;
                        if (depth is <= 40 or >= 150) continue;
                        Cv2.Line(camera, start, far, Scalar.Green, 2);
                        Cv2.Line(camera, end, far, Scalar.Green, 2);
                        Cv2.Circle(camera, start, 6, Scalar.Red);
                        Cv2.Circle(camera, end, 6, Scalar.Blue);
                        Cv2.Circle(camera, far, 6, Scalar.Green);
                        var vectorBuild = Vector<double>.Build;
                        var vectorA = vectorBuild.DenseOfArray(new[]
                            {(double) far.X - start.X, (double) far.Y - start.Y});
                        var vectorB = vectorBuild.DenseOfArray(new[] {(double) far.X - end.X, (double) far.Y - end.Y});
                        var ds = vectorA * vectorB;
                        if (ds / (vectorA.L1Norm() * vectorB.L1Norm()) > 0) acuteAngle++;
                    }

                    if (acuteAngle < 1 && Cv2.ArcLength(t, false) > 600)
                        Cv2.PutText(camera, "Rock", t[0], HersheyFonts.HersheyPlain, 2, Scalar.Red);
                    else if (acuteAngle == 1)
                        Cv2.PutText(camera, "Scissors", t[0], HersheyFonts.HersheyPlain, 2, Scalar.Red);
                    else if (acuteAngle >= 3)
                        Cv2.PutText(camera, "Paper", t[0], HersheyFonts.HersheyPlain, 2, Scalar.Red);
                    else if (Cv2.ArcLength(t, false) > 1000)
                        Cv2.PutText(camera, "Unknown", t[0], HersheyFonts.HersheyPlain, 2, Scalar.Red);
                    acuteAngle = 0;
                }


                Cv2.DrawContours(camera, hull, -1, Scalar.Red);
                Cv2.DrawContours(camera, contours, -1, Scalar.Blue);
                var cameraWindow = new Window("Final", camera, WindowFlags.AutoSize);
                var backgroundWindow = new Window("Skin", skin, WindowFlags.AutoSize);
                Cv2.WaitKey(10);
                if (i >= 20)
                {
                    i = 0;
                    GC.Collect();
                }

                hull.Clear();
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
            Cv2.FindContours(input, out var contours, out var hierarchyIndices, RetrievalModes.External,
                ContourApproximationModes.ApproxSimple);
            return contours;
        }
    }
}