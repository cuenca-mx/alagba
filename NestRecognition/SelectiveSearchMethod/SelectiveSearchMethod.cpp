//
// Created by ricardo_8990 on 8/19/15.
//

#include "SelectiveSearchMethod.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc.hpp>

using namespace ssm;
using namespace cv;
using namespace std;

SelectiveSearchMethod::SelectiveSearchMethod(Mat inputImage, float sigma, int k, int minSize)
{
    //Change colour space to HSV
    Mat image2process;
    float imageSize = inputImage.cols * inputImage.rows;

    cvtColor(inputImage, image2process, CV_RGB2HSV, 0);

    //Convert image to the image format used by the function
    image<rgb> *imageFormat = cvtMatToImage(image2process);

    //Obtain initial regions
    int numCcs;
    double *initialRegions = segment_image(imageFormat, sigma, k, minSize, &numCcs);
    Neighbourhood *neighbourhood = new Neighbourhood();
    regions = universeToRegions(initialRegions, inputImage.cols, inputImage.rows, neighbourhood);

    //Calculate histograms for each regions
    calculateHistograms(image2process);

    //Similarity Set S = 0
    std::vector<similarity> similarities;

    //Calculate the similarity between each region and its neighbourhoods
    for(unordered_map<int, set<int>>::iterator it = neighbourhood->neighbours.begin(); it != neighbourhood->neighbours.end(); ++it)
    {
        int classId = it->first;
        set<int> neighbours = it->second;
        calculateSimilarities(imageSize, similarities, classId, neighbours);
    }
    //Sort elements
    std::sort(similarities.begin(), similarities.end());

    //Iterate to join regions
    while (!similarities.empty() && regions.size() <= 2000)
    {
        similarity maxSim = similarities.back();
        numCcs++;
        //Merge regions
        Region newRegion = mergeRegions(maxSim.regionA, maxSim.regionB);
        regions[numCcs] = newRegion;
        //Remove similarities regarding to first region and second region
        for(vector<similarity>::iterator it = similarities.begin(); it != similarities.end(); )
        {
            if(*it == maxSim)
                it = similarities.erase(it);
            else
                ++it;
        }

        //Update neighbourhood
        set<int> neighA = neighbourhood->getElement(maxSim.regionA);
        set<int> neighB = neighbourhood->getElement(maxSim.regionB);
        set<int> neigh;
        neigh.insert(neighA.begin(), neighA.end());
        neigh.insert(neighB.begin(), neighB.end());
        neigh.erase(maxSim.regionB);
        neigh.erase(maxSim.regionA);
        //Delete neighbourhoods to save memory
        neighbourhood->deleteElement(maxSim.regionA);
        neighbourhood->deleteElement(maxSim.regionB);

        neighbourhood->neighbours[numCcs].insert(neigh.begin(), neigh.end());
        //Calculate similarity with neighbourhoods
        calculateSimilarities(imageSize, similarities, numCcs, neighbourhood->neighbours[numCcs]);

        std::sort(similarities.begin(), similarities.end());
    }

    delete neighbourhood;
    delete initialRegions;
    delete imageFormat;
    image2process.release();
    std::vector<similarity>().swap(similarities);
    globalIt = regions.begin();
}

Rect SelectiveSearchMethod::getProposedRegion()
{
    cv::Rect element = cv::Rect();
    do
    {
        if(globalIt == regions.end())
            return cv::Rect();

        Region region = globalIt->second;
        element = cv::Rect(region.left, region.top, region.right - region.left, region.bottom - region.top);
        ++globalIt;
    } while(element.width < 50 || element.height < 50 || element.width > 347 || element.height > 429);

    return element;
}

image<rgb> *SelectiveSearchMethod::cvtMatToImage(cv::Mat inputImage)
{
    int noCols = inputImage.cols;
    int noRows = inputImage.rows;

    image<rgb> *output = new image<rgb>(noCols, noRows);

    for (int r = 0; r < noRows; ++r)
    {
        for (int s = 0; s < noCols; ++s)
        {
            rgb pixel;
            Vec3b pixelImage = inputImage.at<Vec3b>(r, s);

            pixel.b = pixelImage.val[0];
            pixel.g = pixelImage.val[1];
            pixel.r = pixelImage.val[2];

            output->data[r * noCols + s] = pixel;
        }
    }

    return output;
}

unordered_map<int, Region> SelectiveSearchMethod::universeToRegions(double *segIndices, int width, int height, Neighbourhood* neighbourhood)
{
    unordered_map<int, Region> output;
    unordered_map<double, int> correspondences;
    int currentClass = 1;

    for (int r = 0; r < height; ++r)
    {
        for (int s = 0; s < width; ++s)
        {
            double regionId = segIndices[s * height + r];
            int classId = correspondences[regionId];
            if(classId == 0)
            {
                correspondences[regionId] = currentClass;
                classId = currentClass;
                currentClass++;
            }
            output[classId].insertPoint(s, r);

            //Obtain neighbours
            //Up
            if(r > 0 && correspondences[segIndices[s * height + (r - 1)]] != classId)
            {
                neighbourhood->addRelation(correspondences[segIndices[s * height + (r - 1)]], classId);
            }
            //Left
            if(s > 0 && correspondences[segIndices[(s - 1) * height + r]] != classId)
            {
                neighbourhood->addRelation(correspondences[segIndices[(s - 1) * height + r]], classId);
            }
        }
    }
    correspondences.clear();
    return output;
}

float SelectiveSearchMethod::getSimilarity(int a, int b, float sizeImage)
{
    Region regionA = regions[a];
    Region regionB = regions[b];

    //Get colour similarity
    float histSim = regionA.histogram.getSimilarity(regionB.histogram);

    //Get size similarity
    unsigned long sizeA = getSizePoints(a);
    unsigned long sizeB = getSizePoints(b);
    float sizeSim = 1 - ((sizeA + sizeB) / sizeImage);

    //Get fill similarity
    int right = regionA.right > regionB.right ? regionA.right : regionB.right;
    int left = regionA.left < regionB.left ? regionA.left : regionB.left;
    int top = regionA.top < regionB.top ? regionA.top : regionB.top;
    int bottom = regionA.bottom > regionB.bottom ? regionA.bottom : regionB.bottom;
    int sizeBB = (right - left) * (bottom - top);
    float fillSim = 1 - ((sizeBB - sizeA - sizeB)/ sizeImage);

    return histSim + sizeSim + fillSim;
}

void SelectiveSearchMethod::calculateHistograms(Mat inputImage)
{
    for(unordered_map<int, Region>::iterator r = regions.begin(); r != regions.end(); ++r)
    {
        vector<Point> points = getPointsFromRegion(r->first);
        for(vector<Point>::iterator s = points.begin(); s != points.end(); ++s)
        {
            Vec3b pixel = inputImage.at<Vec3b>(*s);
            r->second.histogram.addValue(pixel);
        }
        //Normalize
        r->second.histogram.normalize();
    }
}

void SelectiveSearchMethod::calculateSimilarities(float imageSize,
                                                  vector<similarity> &similarities, int classId,
                                                  const set<int> &neighbours)
{
    for(set<int>::iterator its = neighbours.begin(); its != neighbours.end(); ++its)
    {
        float simValue = getSimilarity(*its, classId, imageSize);
        similarity s(*its, classId, simValue);
        similarities.push_back(s);
    }
}

Region SelectiveSearchMethod::mergeRegions(int a, int b)
{
    Region output = Region();
    Region regionA = regions[a];
    Region regionB = regions[b];

    //Merge points
    output.classes = pair<int, int>(a, b);

    //Set bounding box
    output.left = regionA.left < regionB.left ? regionA.left : regionB.left;
    output.top = regionA.top < regionB.top ? regionA.top : regionB.top;
    output.right = regionA.right > regionB.right ? regionA.right : regionB.right;
    output.bottom = regionA.bottom > regionB.bottom ? regionA.bottom : regionB.bottom;

    //Update histogram
    output.histogram.mergeHistogram(regionA.histogram, getSizePoints(a), regionB.histogram, getSizePoints(b));
    //Delete histograms to save memory
    regionA.deleteHistogram();
    regionB.deleteHistogram();

    return output;
}

unsigned long SelectiveSearchMethod::getSizePoints(int classId)
{
    Region region = regions[classId];
    if(region.points.empty() && region.classes.first == -1)
        return 0;
    if(!region.points.empty())
        return region.points.size();
    return getSizePoints(region.classes.first) + getSizePoints(region.classes.second);
}

std::vector<cv::Point> SelectiveSearchMethod::getPointsFromRegion(int classId)
{
    Region region = regions[classId];
    if(!region.points.empty())
        return region.points;
    vector<Point> output;
    vector<Point> a = getPointsFromRegion(region.classes.first);
    output.insert(output.end(), a.begin(), a.end());
    a = getPointsFromRegion(region.classes.second);
    output.insert(output.end(), a.begin(), a.end());
    vector<Point>().swap(a);
    return output;
}
