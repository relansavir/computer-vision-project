#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <tuple>
#include <cmath>

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

    int channels[] = { 0, 2 };
    int histSize[] = { 180, 256 };
    float hueRange[] = { 0, 180 };
    float satRange[] = { 0, 256 };
    const float* ranges[] = { hueRange, satRange };
    Mat histogram;
    calcHist(&hls, 1, channels, Mat(), histogram, 2, histSize, ranges);
    GaussianBlur(histogram, histogram, Size(5, 5), 0);

    Point peak;
    minMaxLoc(histogram, nullptr, nullptr, nullptr, &peak);
    int feltHue = peak.y;
    int feltSat = peak.x;

    Mat lightness;
    extractChannel(hls, lightness, 1);
    vector<uchar> values(lightness.begin<uchar>(), lightness.end<uchar>());
    nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    int feltLight = values[values.size() / 2];

    Mat distance(hls.size(), CV_8UC1);
    for (int y = 0; y < hls.rows; y++)
    {
        for (int x = 0; x < hls.cols; x++)
        {
            Vec3b p = hls.at<Vec3b>(y, x);
            int dh = abs(p[0] - feltHue);
            dh = min(dh, 180 - dh);
            int ds = p[2] - feltSat;
            int dl = max(feltLight - (int)p[1], 0);
            double d = sqrt((3.0 * dh) * (3.0 * dh) + (double)(ds * ds) + (1.5 * dl) * (1.5 * dl));            
            distance.at<uchar>(y, x) = (uchar)min(d, 255.0);
        }
    }

    Mat greenMask;
    double otsuThreshold = threshold(distance, greenMask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    cout << "Felt hue " << feltHue << ", saturation " << feltSat<< ", lightness " << feltLight << ", Otsu threshold " << (int)otsuThreshold << endl;
    greenMask.setTo(0, lightness < 25);
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


const int ALLBALLS_GT[16][2] = {
    {253, 908}, {250, 959}, {255, 1013}, {256, 1072}, {254, 1132}, {262, 1184},
    {256, 1244}, {253, 1307}, {329, 901}, {330, 961}, {335, 1013}, {336, 1067},
    {334, 1130}, {345, 1187}, {342, 1245}, {293, 797}
};

const int WHITE_LIGHTNESS = 160;
const double WHITE_BALL_FRACTION = 0.88;
const double APPEARANCE_PENALTY = 0.3;


struct BallDescription
{
    Mat histogram;
    double whiteFraction = 0.0;
};

BallDescription describeBall(const Mat& hls, const Mat& feltMask, Point2f centre)
{
    Mat region = Mat::zeros(hls.size(), CV_8UC1);
    ellipse(region, Point(cvRound(centre.x), cvRound(centre.y)), Size(20, 30),
            0, 0, 360, Scalar(255), FILLED);
    Mat notFelt;
    bitwise_not(feltMask, notFelt);
    bitwise_and(region, notFelt, region);

    BallDescription d;

    int channels[] = { 0, 1, 2 };
    int histSize[] = { 12, 4, 4 };
    float hueRange[] = { 0, 180 };
    float otherRange[] = { 0, 256 };
    const float* ranges[] = { hueRange, otherRange, otherRange };
    calcHist(&hls, 1, channels, region, d.histogram, 3, histSize, ranges);
    normalize(d.histogram, d.histogram, 1.0, 0.0, NORM_L1);

    Mat lightness;
    extractChannel(hls, lightness, 1);
    Mat whitePixels = (lightness > WHITE_LIGHTNESS) & region;
    int total = countNonZero(region);
    d.whiteFraction = (total > 0) ? countNonZero(whitePixels) / (double)total : 0.0;

    return d;
}

struct BallModel
{
    vector<BallDescription> reference;
    double stripeThreshold = 0.0;
};

string ballName(int index);

BallModel trainBallModel(const Mat& allBallsImage){
    BallModel model;
    Mat image = transformImage(allBallsImage);    
    Mat hls;
    cvtColor(image, hls, COLOR_BGR2HLS);
    Mat feltMask = createGreenMask(image);

    for (int b = 0; b < 16; b++)
    {
        Point2f centre((float)ALLBALLS_GT[b][1], (float)ALLBALLS_GT[b][0]);
        model.reference.push_back(describeBall(hls, feltMask, centre));
    }

    double maxSolid = 0.0, minStripe = 1.0;
    for (int b = 0; b < 8; b++)
    {
        maxSolid = max(maxSolid, model.reference[b].whiteFraction);
    }
    for (int b = 8; b < 15; b++)
    {
        minStripe = min(minStripe, model.reference[b].whiteFraction);
    }
    model.stripeThreshold = sqrt(maxSolid * minStripe);

    cout << "Stripe threshold (from AllBalls): " << model.stripeThreshold << endl;
    Mat display = image.clone();
    for (int b = 0; b < 16; b++)
    {
        Point centre(ALLBALLS_GT[b][1], ALLBALLS_GT[b][0]);
        ellipse(display, centre, Size(20, 30), 0, 0, 360, Scalar(0, 255, 255), 2);
        putText(display, ballName(b), Point(centre.x - 10, centre.y - 38),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    }
    imwrite("results/AllBalls_references.jpeg", display);
    return model;
}

double labelCost(const BallDescription& ball, int b, const BallModel& model)
{
    double cost = compareHist(ball.histogram, model.reference[b].histogram,
                              HISTCMP_BHATTACHARYYA);

    bool looksWhite = ball.whiteFraction > WHITE_BALL_FRACTION;
    bool looksStripe = ball.whiteFraction > model.stripeThreshold;
    if (b == 15)
    {
        if (!looksWhite) cost += APPEARANCE_PENALTY;
    }
    else
    {
        if (looksWhite) cost += APPEARANCE_PENALTY;
        if (looksStripe != (b >= 8)) cost += APPEARANCE_PENALTY;
    }
    return cost;
}

vector<int> assignBallLabels(const vector<BallDescription>& balls, const BallModel& model)
{
    vector<tuple<double, int, int>> options;
    for (size_t i = 0; i < balls.size(); i++)
    {
        for (int b = 0; b < 16; b++)
        {
            options.push_back(make_tuple(labelCost(balls[i], b, model), (int)i, b));
        }
    }
    sort(options.begin(), options.end());

    vector<int> labels(balls.size(), -1);
    vector<bool> numberUsed(16, false);
    for (size_t o = 0; o < options.size(); o++)
    {
        int i = get<1>(options[o]);
        int b = get<2>(options[o]);
        if (labels[i] == -1 && !numberUsed[b])
        {
            labels[i] = b;
            numberUsed[b] = true;
        }
    }
    return labels;
}

string ballName(int index)
{
    if (index < 0) return "?";
    return (index == 15) ? "W" : to_string(index + 1);
}

const int GROUND_TRUTH[6][16][2] = {
    { {309, 1092}, {429, 1212}, {383, 1131}, {335, 1050}, {285, 1211}, {261, 1170}, {352, 1173}, {331, 1132}, {240, 1210}, {310, 1170}, {407, 1169}, {281, 1135}, {374, 1212}, {360, 1091}, {331, 1213}, {288, 29} },
    { {282, 1103}, {620, 1224}, {391, 1121}, {315, 1046}, {310, 1305}, {202, 621}, {361, 1173}, {331, 1132}, {177, 725}, {185, 1269}, {665, 798}, {296, 438}, {53, 1235}, {362, 1090}, {399, 1205}, {507, 1169} },
    { {23, 908}, {616, 1224}, {447, 1199}, {336, 29}, {305, 1406}, {210, 621}, {260, 1294}, {226, 1252}, {177, 725}, {171, 1270}, {665, 798}, {249, 351}, {53, 1233}, {169, 901}, {399, 1201}, {434, 29} },
    { {23, 908}, {546, 1407}, {444, 1204}, {-1, -1}, {305, 1406}, {210, 621}, {260, 1294}, {226, 1252}, {181, 724}, {171, 1270}, {665, 798}, {249, 351}, {58, 1231}, {169, 901}, {399, 1201}, {88, 1392} },
    { {535, 23}, {546, 1407}, {444, 1204}, {-1, -1}, {305, 1406}, {131, 25}, {260, 1294}, {226, 1252}, {538, 72}, {171, 1270}, {665, 798}, {249, 351}, {-1, -1}, {169, 901}, {399, 1201}, {608, 26} },
    { {535, 23}, {477, 1364}, {444, 1204}, {-1, -1}, {305, 1406}, {131, 25}, {260, 1294}, {226, 1252}, {538, 72}, {171, 1270}, {-1, -1}, {249, 351}, {-1, -1}, {169, 901}, {399, 1201}, {629, 947} }
};

const double MATCH_DISTANCE = 21.5;

struct DetectionScore
{
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double totalError = 0.0;  
    vector<pair<int, int>> matches; 
};

DetectionScore evaluateDetection(const vector<Point2f>& detections, int tableIndex)
{
    vector<Point2f> truth;
    vector<int> truthId;
    for (int b = 0; b < 16; b++)
    {
        int row = GROUND_TRUTH[tableIndex][b][0];
        int col = GROUND_TRUTH[tableIndex][b][1];
        if (row >= 0)
        {
            truth.push_back(Point2f((float)col, (float)row));
            truthId.push_back(b);
        }
    }

    // Every possible (detection, truth) pair, sorted by distance
    vector<tuple<double, int, int>> pairs;
    for (size_t d = 0; d < detections.size(); d++)
    {
        for (size_t g = 0; g < truth.size(); g++)
        {
            pairs.push_back(make_tuple(norm(detections[d] - truth[g]), (int)d, (int)g));
        }
    }
    sort(pairs.begin(), pairs.end());

    vector<bool> detectionUsed(detections.size(), false);
    vector<bool> truthUsed(truth.size(), false);
    DetectionScore score;

    for (size_t p = 0; p < pairs.size(); p++)
    {
        double distance = get<0>(pairs[p]);
        int d = get<1>(pairs[p]);
        int g = get<2>(pairs[p]);

        if (distance > MATCH_DISTANCE)
        {
            break;
        }
        if (detectionUsed[d] || truthUsed[g])
        {
            continue;
        }

        detectionUsed[d] = true;
        truthUsed[g] = true;
        score.truePositives++;
        score.totalError += distance;
        score.matches.push_back(make_pair(d, truthId[g]));
    }

    score.falsePositives = (int)detections.size() - score.truePositives;
    score.falseNegatives = (int)truth.size() - score.truePositives;
    return score;
}

 
void printScore(const string& label, const DetectionScore& s)
{
    double precision = s.truePositives / (double)(s.truePositives + s.falsePositives);
    double recall = s.truePositives / (double)(s.truePositives + s.falseNegatives);
    double f1 = 2 * precision * recall / (precision + recall);
    double meanError = s.truePositives > 0 ? s.totalError / s.truePositives : 0.0;
 
    cout << label << "  TP=" << s.truePositives << " FP=" << s.falsePositives
         << " FN=" << s.falseNegatives << fixed << setprecision(3)
         << "  Precision=" << precision << " Recall=" << recall
         << " F1=" << f1 << setprecision(1)
         << "  Mean error=" << meanError << " mm" << endl;
    cout.unsetf(ios::fixed);
}

void MyApplication()
{
    const char* file_location = "data/";
    const char* pool_files[] = {
        "AllBalls.jpeg",
        "Table1.jpeg",
        "Table2.jpeg",
        "Table3.jpeg",
        "Table4.jpeg",
        "Table5.jpeg",
        "Table6.jpeg"
    };

    int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);
    Mat* pool_image = new Mat[number_of_images];
    for (int file_no = 0; file_no < number_of_images; file_no++)
    {
        string filename(file_location);
        filename.append(pool_files[file_no]);
        pool_image[file_no] = imread(filename);
        if (pool_image[file_no].empty())
        {
            cout << "Could not open " << filename << endl;
            delete[] pool_image;
            return;
        }
    }

    BallModel model = trainBallModel(pool_image[0]);

    int confusion[16][16] = { 0 };
    int unlabelled = 0;
    DetectionScore total;

    for (int file_no = 1; file_no < number_of_images; file_no++)
    {
        int table = file_no;
        cout << "\n" << pool_files[file_no] << endl;

        Mat image = transformImage(pool_image[file_no]);

        Mat greenMask = createGreenMask(image);
        Mat cushionMask = createCushionMask(image);
        Mat backgroundMask;
        bitwise_or(greenMask, cushionMask, backgroundMask);

        vector<vector<Point>> ballContours = findBallContours(backgroundMask);
        vector<Point2f> ballCentres = findBallCentres(ballContours, image.size());

        DetectionScore score = evaluateDetection(ballCentres, table - 1);
        printScore("Table " + to_string(table), score);
        total.truePositives += score.truePositives;
        total.falsePositives += score.falsePositives;
        total.falseNegatives += score.falseNegatives;
        total.totalError += score.totalError;

        Mat hls;
        cvtColor(image, hls, COLOR_BGR2HLS);
        vector<BallDescription> balls;
        for (size_t i = 0; i < ballCentres.size(); i++)
        {
            balls.push_back(describeBall(hls, greenMask, ballCentres[i]));
        }
        vector<int> labels = assignBallLabels(balls, model);

        for (size_t m = 0; m < score.matches.size(); m++)
        {
            int trueBall = score.matches[m].second;
            int predicted = labels[score.matches[m].first];
            if (predicted < 0)
            {
                unlabelled++;
                continue;
            }
            confusion[trueBall][predicted]++;
        }

        Mat result = image.clone();
        drawContours(result, ballContours, -1, Scalar(0, 0, 255), 3);
        for (size_t i = 0; i < ballCentres.size(); i++)
        {
            drawMarker(result, ballCentres[i], Scalar(255, 0, 0), MARKER_CROSS, 30, 3);
            putText(result, ballName(labels[i]),
                    Point((int)ballCentres[i].x + 15, (int)ballCentres[i].y - 15),
                    FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2);
            cout << "  Ball " << ballName(labels[i]) << ": column "
                 << (int)ballCentres[i].x << ", row " << (int)ballCentres[i].y << endl;
        }

        string prefix = "results/detection/Table" + to_string(table);
        imwrite(prefix + "_1_transformed.jpeg", image);
        imwrite(prefix + "_2_background_mask.png", backgroundMask);
        imwrite(prefix + "_3_detections.jpeg", result);

        imshow(pool_files[file_no], result);
        waitKey(0);
        destroyAllWindows();
    }

    printScore("OVERALL", total);

    int correct = 0, evaluated = unlabelled;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            evaluated += confusion[i][j];
        }
        correct += confusion[i][i];
    }
    int totalBalls = total.truePositives + total.falseNegatives;
    cout << "\nRECOGNITION  correct=" << correct << " of " << evaluated
         << " detected balls  accuracy=" << fixed << setprecision(3)
         << correct / (double)evaluated << endl;
    cout << "END-TO-END  correct=" << correct << " of " << totalBalls
         << " balls  accuracy=" << correct / (double)totalBalls << endl;
    cout.unsetf(ios::fixed);

    cout << "\nConfusion matrix (rows = true ball, columns = predicted)\n    ";
    for (int j = 0; j < 16; j++)
    {
        cout << setw(3) << ballName(j);
    }
    cout << endl;
    for (int i = 0; i < 16; i++)
    {
        cout << setw(3) << ballName(i) << " ";
        for (int j = 0; j < 16; j++)
        {
            cout << setw(3) << confusion[i][j];
        }
        cout << endl;
    }

    delete[] pool_image;
}

int main()
{
    MyApplication();
    return 0;
}