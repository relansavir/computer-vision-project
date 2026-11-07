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
    Mat hls;
    cvtColor(image, hls, COLOR_BGR2HLS);

    Mat greenMask;
    Scalar lowerGreen(46, 25, 4);
    Scalar upperGreen(70, 255, 94);

    inRange(hls, lowerGreen, upperGreen, greenMask);

    return greenMask;
}

Mat createCushionMask(const Mat& image)
{
    Mat hls;
    cvtColor(image, hls, COLOR_BGR2HLS);
    Mat cushionColour;
    inRange(hls, Scalar(38, 25, 20), Scalar(85, 190, 255), cushionColour);

    int border = 45;
    Mat borderStrip(image.size(), CV_8UC1, Scalar(255));
    borderStrip(Rect(border, border,
                     image.cols - 2 * border,
                     image.rows - 2 * border)).setTo(0);

    Mat cushionMask;
    bitwise_and(cushionColour, borderStrip, cushionMask);

    return cushionMask;
}

vector<vector<Point>> findBallContours(const Mat& greenMask)
{
    Mat ballMask;
    bitwise_not(greenMask, ballMask);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    morphologyEx(ballMask, ballMask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(ballMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> ballContours;
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contourArea(contours[i]) >= 1000)
        {
            ballContours.push_back(contours[i]);
        }
    }

    return ballContours;
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
    Mat cushionMask = createCushionMask(image);
    Mat backgroundMask;
    bitwise_or(greenMask, cushionMask, backgroundMask);

    vector<vector<Point>> ballContours = findBallContours(backgroundMask);
    cout << "Regions found: " << ballContours.size() << endl;
    Mat result = image.clone();
    drawContours(result, ballContours, -1, Scalar(0, 0, 255), 3);
    imshow("Original", image);
    imshow("Green Mask", greenMask);
    imshow("Detected Regions", result);
 
    waitKey(0);
    destroyAllWindows();
}

int main()
{
    MyApplication();
    return 0;
}