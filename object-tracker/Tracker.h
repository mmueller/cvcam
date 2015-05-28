#ifndef __CVCAM_TRACKER_H__
#define __CVCAM_TRACKER_H__

#include <list>
#include <opencv2/opencv.hpp>

typedef std::list<cv::Mat> MatList;

struct TrackerResult {
    // Confidence in this result: 0.0 (least confident) - 1.0 (most confident)
    double confidence; // UNUSED

    // Bounding box of target, if present (NULL otherwise)
    cv::Ptr<cv::Rect> boundingBox;

    // Color of bounding box (for debugging only)
    cv::Scalar color;

    TrackerResult() :
        confidence(0.0),
        color(0, 0, 0)
        { }
};

class Tracker {
    public:
        Tracker();
        ~Tracker();

        TrackerResult track(const cv::Mat &inFrame);

    private:

        void initBackground();
        void updateBackground(const cv::Mat &mask);
        void updateNoiseLevel();
        cv::Mat findDeviations();

        MatList m_recentFrames;
        MatList m_recentBackgrounds;
        double m_noiseLevel;
};

#endif //__CVCAM_TRACKER_H__
