//
// Created by ricardo_8990 on 8/19/15.
// Class used for the Selective Search algorithm, as it is not implemented in the OpenCV library, this class is
// separated such that it can be aported to the community. However, it has to be adapted to the code rules used by
// the OpenCV community.
//

#ifndef TRACKING_SELECTIVESEARCHMETHOD_H
#define TRACKING_SELECTIVESEARCHMETHOD_H

#include <opencv2/core.hpp>
#include "segment/image.h"
#include "segment/misc.h"
#include "segment/segment-image.h"
#include <limits>
#include <unordered_map>
#include <set>
#include <boost/ptr_container/ptr_vector.hpp>

namespace ssm
{
    class Neighbourhood
    {
    public:

        std::unordered_map<int, std::set<int>> neighbours;

        inline ~Neighbourhood()
        {
            neighbours.clear();
        }

        inline void addRelation(int a, int b)
        {
            if( a > b )
                neighbours[a].insert(b);
            else
                neighbours[b].insert(a);
        }

        inline std::set<int> getElement(int classId)
        {
            return neighbours[classId];
        }

        inline void deleteElement(int classId)
        {
            neighbours.erase(classId);
        }
    };

    class Histogram
    {
    public:

        inline Histogram()
        {
            total = std::vector<float>(bins * 3, 0);
        }

        inline void addValue(cv::Vec3b pixel)
        {
            //Update h channel
            int position = int((pixel.val[0] * bins) / hRanges);
            total[position] += 1;
            //Update s channel
            position = int((pixel.val[1] * bins) / sRanges);
            total[position + bins] += 1;
            //Update v channel
            position = int((pixel.val[2] * bins) / sRanges);
            total[position + bins + bins] += 1;
        }

        inline void normalize()
        {
            cv::normalize(total, total, 1, 0, cv::NORM_L1, -1, cv::Mat());
        }

        inline float getSimilarity(Histogram b)
        {
            float output = 0;
            for(int r = 0; r < bins; ++r)
            {
                output += std::min(total[r], b.total[r]);
            }
            return output;
        }

        inline void mergeHistogram(Histogram a, unsigned long sizeA, Histogram b, unsigned long sizeB)
        {
            total = std::vector<float>(bins * 3, 0);
            for(int r = 0; r < bins * 3; ++r)
            {
                float newValue = ((sizeA * a.total[r]) + (sizeB * b.total[r])) / (sizeA + sizeB);
                total[r] = newValue;
            }
        }

        inline void deleteTotal()
        {
            std::vector<float>().swap(total);
        }

    private:
        std::vector<float> total;
        static constexpr int bins = 25;
        static constexpr float hRanges = 180;
        static constexpr float sRanges = 256;
    };

    class Region
    {
    public:
        int left;
        int top;
        int right;
        int bottom;
        Histogram histogram;
        std::vector<cv::Point> points;
        std::pair<int, int> classes;

        inline ~Region()
        {
            std::vector<cv::Point>().swap(points);
        }

        inline Region()
        {
            left = std::numeric_limits<int>::max();
            right = 0;
            top = std::numeric_limits<int>::max();
            bottom = 0;
            histogram = Histogram();
            classes = std::pair<int, int>(-1, -1);
        }

        inline void insertPoint(int x, int y)
        {
            cv::Point_<int> p(x, y);
            if(x < left)
                left = x;
            if(x > right)
                right = x;
            if(y < top)
                top = y;
            if(y > bottom)
                bottom = y;

            points.push_back(p);
        }

        inline void deleteHistogram()
        {
            histogram.deleteTotal();
        }
    };

    class SelectiveSearchMethod
    {
    public:

        SelectiveSearchMethod(cv::Mat inputImage, float sigma, int k, int minSize);
        cv::Rect getProposedRegion();
        inline void clear()
        {
            regions.clear();
        }

        struct similarity{
            int regionA;
            int regionB;
            float simValue;

            inline similarity(int regionA, int regionB, float simValue)
            {
                this->regionA = regionA;
                this->regionB = regionB;
                this->simValue = simValue;
            }

            bool operator < (const similarity& i) const
            {
                return simValue < i.simValue;
            }

            bool operator == (const similarity& a) const
            {
                return a.regionA == regionA || a.regionA == regionB || a.regionB == regionA || a.regionB == regionB;
            }
        };

    private:
        std::unordered_map<int, Region> regions;
        std::unordered_map<int, Region>::iterator globalIt;

    private:
        image<rgb> *cvtMatToImage(cv::Mat inputImage);

        std::unordered_map<int, Region> universeToRegions(double *segIndices, int width, int height, Neighbourhood* neighbourhood);

        float getSimilarity(int a, int b, float sizeImage);

        void calculateHistograms(cv::Mat inputImage);

        Region mergeRegions(int a, int b);

        void calculateSimilarities(float imageSize, std::vector<similarity> &similarities, int classId,
                                   const std::set<int> &neighbours);

        unsigned long getSizePoints(int classId);

        std::vector<cv::Point> getPointsFromRegion(int classId);
    };
}

#endif //TRACKING_SELECTIVESEARCHMETHOD_H
