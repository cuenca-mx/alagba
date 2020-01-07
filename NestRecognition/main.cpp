/*
 * The University of Nottingham
 * School of Computer Science
 *
 * Ricardo Sánchez Castillo
 * Student ID 4225015
 *
 * Program used for recognise nests within an image.
 * It requires the OpenCV and Caffe library which can be installed from:
 * http://caffe.berkeleyvision.org/installation.html
 * http://opencv.org/downloads.html
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <dirent.h>
#include "Classifier.h"

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace caffe;

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
    << "architecture. The imageFolder contaning the images to recognise and "           << endl
    << "the Results folder name where the images are going to be stored."
    << "Usage:"                                                                         << endl
    << "./NestRecognition deploy.prototxt weights.caffemodel ImageFolder Results"       << endl
    << "------------------------------------------------------------------------------" << endl
    << endl;
}

int main(int argc, char** argv)
{
    help();
    //Verify parameters
    if(argc != 5)
    {
        cerr << "Error in parameters" << endl;
        return 1;
    }

    //Load parameters
    string model = argv[1];
    string weights = argv[2];
    string imageDir = argv[3];
    string results = argv[4];
    Mat image;

    //Create classifier
    Classifier classifier(model, weights);
    ofstream newFile;
    newFile.open(imageDir + results +  "/nohup.out");
    int i = 0;
    int totalPositives = 0;
    int totalNegatives = 0;
    int totalTime = 0;

    DIR* dir;
    struct dirent *ent;
    if( (dir = opendir(imageDir.c_str())) != NULL )
    {
        while ((ent = readdir(dir)) != NULL)
        {
            String imageFile = imageDir + ent->d_name;

            //Load image
            image = imread(imageFile, 1);

            if(!image.data)
            {
                continue;
            }

            i++;
            newFile << "Image " << i << "\n";

            //Classify
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            int result = classifier.Classify(image);
            high_resolution_clock::time_point t2 = high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::minutes>(t2 - t1).count();
            newFile << "Time taken: " << duration << " \n";

            //Add rectangles to the image
            int negatives = 0;
            int positives = 0;
            for(vector<Nest>::iterator it = classifier.nests.begin(); it != classifier.nests.end(); ++it)
            {
                Scalar colour;
                if(it->probability >= 0.5)
                {
                    colour = Scalar(0, 255, 0);
                    positives++;
                }
                else
                {
                    colour = Scalar(255, 0, 0);
                    negatives++;
                }
                rectangle(image, it->rect, colour, 2, LINE_AA, 0);
                putText(image, to_string(it->probability), it->rect.br(), FONT_HERSHEY_COMPLEX, 1, colour, 2, LINE_AA, 0);
            }

            classifier.nests.clear();

            //Print result
            newFile << "Positives " << positives << "\n";
            newFile << "Negatives " << negatives << "\n";
            totalPositives += positives;
            totalNegatives += negatives;
            totalTime += duration;
            String name = imageDir + results + "/" + results + to_string(i) + ".png";
            imwrite(name, image);
        }
    }
    newFile << "Total positives " << totalPositives << "\n";
    newFile << "Total negatives " << totalNegatives << "\n";
    newFile << "Total time " << totalTime << "\n";
    newFile.close();
}