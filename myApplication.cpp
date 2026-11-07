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

bool isNearPocket(Point2f centre)
{
    Point2f pockets[6] = {
        Point2f(0, 0),     Point2f(717, 0),   Point2f(1433, 0),
        Point2f(0, 682),   Point2f(717, 682), Point2f(1433, 682)
    };
 
    for (int i = 0; i < 6; i++)
    {
        if (norm(centre - pockets[i]) < 60)
        {
            return true;
        }
    }
    return false;
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
        if (contourArea(contours[i]) < 1000)
        {
            continue;
        }
 
        Moments m = moments(contours[i]);
        Point2f centre(m.m10 / m.m00, m.m01 / m.m00);
 
        if (isNearPocket(centre))
        {
            continue;
        }
 
        ballContours.push_back(contours[i]);
    }
 
    return ballContours;
}

vector<Point2f> findBallCentres(const vector<vector<Point>>& ballContours,
                                Size imageSize)
{
    const double SINGLE_BALL_AREA = 3000.0;
    const float VERTICAL_STRETCH = 1.5f;

    vector<Point2f> centres;

    for (size_t i = 0; i < ballContours.size(); i++)
    {
        double area = contourArea(ballContours[i]);
        int numBalls = max(1, (int)round(area / SINGLE_BALL_AREA));

        if (numBalls == 1)
        {
            Moments m = moments(ballContours[i]);
            centres.push_back(Point2f(m.m10 / m.m00, m.m01 / m.m00));
            continue;
        }

        Mat regionMask = Mat::zeros(imageSize, CV_8UC1);
        drawContours(regionMask, ballContours, (int)i, Scalar(255), FILLED);

        vector<Point> pixels;
        findNonZero(regionMask, pixels);

        Mat data((int)pixels.size(), 2, CV_32F);
        for (size_t p = 0; p < pixels.size(); p++)
        {
            data.at<float>((int)p, 0) = (float)pixels[p].x;
            data.at<float>((int)p, 1) = (float)pixels[p].y / VERTICAL_STRETCH;
        }

        Mat labels, clusterCentres;
        theRNG().state = 12345;
        kmeans(data, numBalls, labels,
               TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1),
               5, KMEANS_PP_CENTERS, clusterCentres);

        for (int c = 0; c < numBalls; c++)
        {
            float x = clusterCentres.at<float>(c, 0);
            float y = clusterCentres.at<float>(c, 1) * VERTICAL_STRETCH;
            centres.push_back(Point2f(x, y));
        }
    }

    return centres;
}

void MyApplication()
{
        for (int t = 1; t <= 6; t++)
    {
    string filename = "data/Table" + to_string(t) + ".jpeg";
    Mat original = imread(filename);
    if (original.empty())
    {
        cout << "Could not load " << filename << endl;
        continue;
    }
    Mat image = transformImage(original);
    cout << "Table " << t << ":" << endl;
    Mat greenMask = createGreenMask(image);
    Mat cushionMask = createCushionMask(image);
    Mat backgroundMask;
    bitwise_or(greenMask, cushionMask, backgroundMask);

    vector<vector<Point>> ballContours = findBallContours(backgroundMask);
    cout << "Regions found: " << ballContours.size() << endl;
    Mat result = image.clone();
    drawContours(result, ballContours, -1, Scalar(0, 0, 255), 3);
    vector<Point2f> ballCentres = findBallCentres(ballContours, image.size());
    cout << "Balls found: " << ballCentres.size() << endl;

    for (size_t i = 0; i < ballCentres.size(); i++)
    {
        drawMarker(result, ballCentres[i], Scalar(255, 0, 0),
                   MARKER_CROSS, 30, 3);
        cout << "  Ball " << i + 1 << ": column " << (int)ballCentres[i].x
             << ", row " << (int)ballCentres[i].y << endl;
    }
    
    string prefix = "results/detection/Table" + to_string(t);
    imwrite(prefix + "_1_transformed.jpeg", image);
    imwrite(prefix + "_2_background_mask.png", backgroundMask);
    imwrite(prefix + "_3_detections.jpeg", result);
    }
 
    waitKey(0);
    destroyAllWindows();
}

int main()
{
    MyApplication();
    return 0;
}