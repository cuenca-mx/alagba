/*
 * The University of Nottingham
 * School of Computer Science
 *
 * Ricardo Sánchez Castillo
 * Student ID 4225015
 *
 * Program used for tracking video nests in a video stream as described in the dissertation document.
 * It requires the OpenCV and Caffe library which can be installed from:
 * http://caffe.berkeleyvision.org/installation.html
 * http://opencv.org/downloads.html
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Classifier/Classifier.h"

using namespace std;
using namespace cv;

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
    << "This program detects nests within a video stream."                              << endl
    << "It requires the prototxt file and the weights to initialize the caffe "         << endl
    << "architecture."                                                                  << endl
    << "Usage:"                                                                         << endl
    << "./Tracking deploy.prototxt weights.caffemodel InputVideoFile"                   << endl
    << "------------------------------------------------------------------------------" << endl
    << endl;
}

int main(int argc, char** argv)
{
    help();
    namedWindow("Main", WINDOW_NORMAL);
    //Verify parameters
    if(argc != 4)
    {
        cerr << "Error in parameters" << endl;
        return 1;
    }

    //Load parameters
    string model = argv[1];
    string weights = argv[2];
    string videoFile = argv[3];

    //Create classifier
    Classifier classifier(model, weights, ISlideMethod::SELECTIVE);

    //Open video file
    VideoCapture cap(videoFile);

    if(!cap.isOpened())
    {
        cerr << "Could not open video" <<endl;
        return -1;
    }

    //Prepare video file to be rendered
    //Code based from the OpenCV documentation in: http://docs.opencv.org/doc/tutorials/highgui/video-write/video-write.html
    string::size_type nameFile = videoFile.find_last_of('.');
    const string finalNameFile = videoFile.substr(0, nameFile) + "_processed" + ".avi";
    int codecType = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));

    Size frameSize = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo;
    outputVideo.open(finalNameFile, codecType, cap.get(CV_CAP_PROP_FPS), frameSize, true);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << videoFile << endl;
        return -1;
    }

    Mat frame;
    Mat image;
    int currentFrameNumber = 0;
    double totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);

    while(true)
    {
        cout << "Processing frame " << ++currentFrameNumber << " of " << totalFrames << endl;
        cap >> frame;

        if(frame.empty())
            break;

        frame.copyTo(image);
        classifier.Classify(image);

        //Add rectangles to the image
        int i = 0;
        for(vector<Nest>::iterator it = classifier.nests.begin(); it != classifier.nests.end(); ++it)
        {
            Scalar colour;
            if(it->probability >= 0.5)
                colour = Scalar(0, 255, 0);
            else
                colour = Scalar(255, 0, 0);
            rectangle(image, it->rect, colour, 2, LINE_AA, 0);
            putText(image, to_string(it->probability), it->rect.br(), FONT_HERSHEY_COMPLEX, 1, colour, 2, LINE_AA, 0);
        }

        outputVideo << image;
        imshow("Main", image);
        waitKey(30);
    };

    return 0;
}