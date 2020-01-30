/*
 * The University of Nottingham
 * School of Computer Science
 *
 * Ricardo Sánchez Castillo
 * Student ID 4225015
 *
 * Program used for cutting parts of images.
 * It requires the OpenCV and Caffe library which can be installed from:
 * http://caffe.berkeleyvision.org/installation.html
 * http://opencv.org/downloads.html
 */

#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void onTrackBar(int, void *);
void onMouse(int, int, int, int, void *);
void saveImage();

bool existFile(String name);

int angle = 45;
bool isRecording = false;
Point originPoint;
Point endPoint;
Mat currentImage;
String destinationFolder;
String trashFolder;
String baseName = "Nest";
int numberOfNest = 0;

static void help()
{
    cout
    << "------------------------------------------------------------------------------" << endl
    << "The University of Nottingham"                                                   << endl
    << "School of Computer Science"                                                     << endl
    << ""                                                                               << endl
    << "Automatic Sea Turtle Nest Detection via Deep Learning"                          << endl
    << ""                                                                               << endl
    << "Ricardo Sanchez Castillo"                                                       << endl
    << "Student ID: 4225015"                                                            << endl
    << ""                                                                               << endl
    << ""                                                                               << endl
    << "This program is used for cutting the images in order to obtain only the nest."  << endl
    << "It requires the image folder where the images are going to be taken, the "      << endl
    << "folder where the images processed are going to be stored and the folder where " << endl
    << "the results images are going to be stored"                                      << endl
    << "Usage:"                                                                         << endl
    << "./Tijeras originFolder processedFolder resultsFolder"                           << endl
    << "------------------------------------------------------------------------------" << endl
    << endl;
}

int main(int argc, char** argv)
{
    help();
    //Verify parameters
    if(argc != 4)
    {
        cerr << "Error in parameters" << endl;
        return 1;
    }

    //Load parameters
    destinationFolder = argv[1];
    trashFolder = argv[2];
    string dirStr = argv[3];

    //Create the GUI
    namedWindow("Main", WINDOW_NORMAL);
    createTrackbar("Rotation", "Main", &angle, 90, onTrackBar);
    setMouseCallback("Main", onMouse, 0);

    //Get a list of files from a directory
    DIR* dir;
    struct dirent *ent;

    if( (dir = opendir(dirStr.c_str())) != NULL )
    {
        while ( (ent = readdir(dir)) != NULL )
        {
            String name = dirStr + ent->d_name;
            currentImage = imread(name, 1);

            if(!currentImage.data)
            {
                printf("Image could not be read: %s\n", name.c_str());
                continue;
            }

            imshow("Main", currentImage);
            displayStatusBar("Main", ent->d_name, 0);

            while(true)
            {
                int codeKey = waitKey(0);
                if(codeKey == 115)
                {
                    //Save file
                    saveImage();
                }
                if(codeKey == 110)
                {
                    //Move image to processed
                    String newFileName = trashFolder + ent->d_name;
                    rename(name.c_str(), newFileName.c_str());
                    //Next image
                    angle = 45;
                    setTrackbarPos("Rotation", "Main", angle);
                    break;
                }
            }
        }
        closedir(dir);
    }
    else
    {
        printf("Directory not found");
        return -1;
    }

    destroyWindow("Main");
    return 0;
}

void onTrackBar(int, void *)
{
    int rows = currentImage.rows;
    int cols = currentImage.cols;


    Point2f center = cv::Point_<float>(rows / 2, cols / 2);
    Mat matrix = getRotationMatrix2D(center, angle - 45, 1);
    warpAffine(currentImage, currentImage, matrix, currentImage.size());
    imshow("Main", currentImage);
}

void onMouse(int event, int x, int y, int, void*)
{
    if(event == EVENT_LBUTTONDOWN && !isRecording)
    {
        originPoint = Point(x, y);
        isRecording = true;
    }

    if(event == EVENT_MOUSEMOVE && isRecording)
    {
        Mat imageToShow = currentImage.clone();
        rectangle(imageToShow, originPoint, Point(x, y), Scalar(255, 0, 0), 2, LINE_AA);
        imshow("Main", imageToShow);
    }

    if(event == EVENT_LBUTTONUP && isRecording)
    {
        endPoint = Point(x, y);
        isRecording = false;
    }
}

void saveImage()
{
    if(originPoint.x == endPoint.x || originPoint.y == endPoint.y)
        return;
    Mat toSave(currentImage, cv::Rect(originPoint.x < endPoint.x ? originPoint.x : endPoint.x,
                                        originPoint.y < endPoint.y ? originPoint.y : endPoint.y,
                                        abs(originPoint.x - endPoint.x), abs(originPoint.y - endPoint.y)));
    char name[5];
    bool saving = true;
    String fileName;
    String fullFileName;

    while(saving)
    {
        numberOfNest++;
        sprintf(name, "%05d", numberOfNest);
        fileName = baseName + name + ".tif";
        fullFileName = destinationFolder + fileName;
        if(!existFile(fullFileName))
            saving = false;
    }

    imwrite(fullFileName, toSave);
    displayStatusBar("Main", fileName + " created...", 0);
}

bool existFile(String name)
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}