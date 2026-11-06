#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat transformImage(const Mat& image)
{
    // 1. Define the four corners in the original image
    Point2f srcPoints[4] = {
        Point2f(337, 35),    // Top-left
        Point2f(1570, 21),   // Top-right
        Point2f(1888, 451),  // Bottom-right
        Point2f(38, 472)     // Bottom-left
    };

    // 2. Define the corresponding rectangular destination
    Point2f dstPoints[4] = {
        Point2f(0, 0),
        Point2f(1433, 0),
        Point2f(1433, 682),
        Point2f(0, 682)
    };

    // 3. Calculate the perspective transformation matrix
    Mat transformationMatrix =
        getPerspectiveTransform(srcPoints, dstPoints);

    // 4. Apply the transformation
    Mat transformedImage;

    warpPerspective(
        image,
        transformedImage,
        transformationMatrix,
        Size(1434, 683)
    );

    return transformedImage;
}
Mat createGreenMask(const Mat& image)
{
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    Mat greenMask;

    // Initial experimental green range
    Scalar lowerGreen(35, 30, 25);
    Scalar upperGreen(95, 255, 220);

    inRange(hsv, lowerGreen, upperGreen, greenMask);

    return greenMask;
}

void MyApplication()
{
    Mat image = imread("results/Table2_transformed.jpeg");
    if (image.empty())
    {
        cout << "Image not found!" << endl;
        return;
    }

    Mat greenMask = createGreenMask(image);

    imshow("Original", image);
    imshow("Green Mask", greenMask);

    waitKey(0);
    destroyAllWindows();
}
int main()
{
    MyApplication();
    return 0;
}