#include <iostream>

#include "Tracker.h"

#define FRAMES_TO_TRACK 5
#define WINDOW_NAME "Tracker DEBUG"

const bool USE_BLUR = true;
const bool DEBUG_WINDOW = true;

static cv::Mat
getContours(const cv::Mat &inFrame)
{
    cv::Mat canny_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    int thresh1 = 1; // TODO: HUH?
    int thresh2 = 255; // TODO: HUH?

    // Detect edges using canny
    cv::Canny(inFrame, canny_output, thresh1, thresh2);

    // Find contours
    cv::findContours(canny_output, contours, CV_RETR_TREE,
                     CV_CHAIN_APPROX_NONE);

    // Draw contours
    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::drawContours(drawing, contours, -1, color, CV_FILLED);

    return drawing;
}

/* Reads from and writes to a single-channel 8-bit image, returns bounding
 * rectangles for all contours. */
static std::vector<cv::Rect>
fillContours(cv::Mat &inFrame)
{
    cv::Mat canny_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    int thresh1 = 1; // TODO: HUH?
    int thresh2 = 255; // TODO: HUH?

    // Detect edges using canny
    cv::Canny(inFrame, canny_output, thresh1, thresh2);

    // Find contours
    cv::findContours(canny_output, contours, hierarchy, CV_RETR_TREE,
                     CV_CHAIN_APPROX_NONE);

    // Draw contours
    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8U);
    cv::Scalar color = cv::Scalar(255);
    cv::drawContours(inFrame, contours, -1, 255, CV_FILLED);
    std::vector<cv::Rect> bounding_rectangles(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<std::vector<cv::Point> > hulls;
        hulls.push_back(std::vector<cv::Point>());
        convexHull(cv::Mat(contours[i]), hulls[0], false);
        bounding_rectangles[i] = cv::boundingRect(cv::Mat(hulls[0]));
        cv::drawContours(inFrame, hulls, 0, color, CV_FILLED, 8,
                         std::vector<cv::Vec4i>(), 0, cv::Point());
    }
    return bounding_rectangles;
}

static cv::Mat
dilate(const cv::Mat &inFrame, int size)
{
    cv::Mat result;

    cv::Mat element = cv::getStructuringElement(
                            cv::MORPH_ELLIPSE, cv::Size(size, size));
    cv::dilate(inFrame, result, element);

    return result;
}

static cv::Mat
erode(const cv::Mat &inFrame, int size)
{
    cv::Mat result;

    cv::Mat element = cv::getStructuringElement(
                            cv::MORPH_ELLIPSE, cv::Size(size, size));
    cv::erode(inFrame, result, element);

    return result;
}

static cv::Mat
grayout(const cv::Mat &image, const cv::Mat &mask)
{
    int max_x = image.size().width;
    int max_y = image.size().height;
    cv::Mat result = image.clone();
    for (int x = 0; x < max_x; x++) {
        for (int y = 0; y < max_y; y++) {
            cv::Point p = cv::Point(x, y);
            if (mask.at<uchar>(p)) {
                cv::Vec3b values = image.at<cv::Vec3b>(p);
                uchar avg = (values.val[0] + values.val[1] + values.val[2]) / 3;
                values.val[0] = values.val[1] = values.val[2] = avg;
                result.at<cv::Vec3b>(p) = values;
            }
        }
    }
    return result;
}

Tracker::Tracker()
    : m_noiseLevel(441)
{
    if (DEBUG_WINDOW) {
        cv::namedWindow(WINDOW_NAME);
    }
}

Tracker::~Tracker()
{
    if (DEBUG_WINDOW) {
        cv::destroyWindow(WINDOW_NAME);
    }
}

void
Tracker::initBackground()
{
    double frameWeight = 1.0 / FRAMES_TO_TRACK;
    MatList::iterator i = m_recentFrames.begin();
    cv::Mat background;
    i->convertTo(background, CV_32FC3);
    background *= frameWeight;
    for (i++; i != m_recentFrames.end(); ++i) {
        cv::Mat x;
        i->convertTo(x, CV_32FC3);
        cv::addWeighted(background, 1.0, x, frameWeight, 0.0, background);
    }
    m_recentBackgrounds.push_front(background);
}

void
Tracker::updateNoiseLevel()
{
    double delta = 0.0;
    int points = 0;

    MatList::const_iterator f2 = m_recentFrames.begin();
    MatList::const_iterator f1 = f2;
    f2++;

    for (int x = 0; x < f1->size().width; x += 10) {
        for (int y = 0; y < f1->size().height; y += 10) {
            points++;
            cv::Point p(x, y);
            cv::Vec3b p1 = f1->at<cv::Vec3b>(p);
            cv::Vec3b p2 = f2->at<cv::Vec3b>(p);
            delta += cv::norm(p1-p2);
        }
    }

    // Minimum noise level is 10. Because I say so.
    double currentNoise = delta/points;
    if (currentNoise < 10.0) {
        currentNoise = 10.0;
    }

    m_noiseLevel = (m_noiseLevel * 0.9) + (currentNoise * 0.1);
}

void
Tracker::updateBackground(const cv::Mat &mask)
{
    double newFrameWeight = 0.1;
    double frameWeight = (1.0 - newFrameWeight) / m_recentBackgrounds.size();

    // Add current frame with foreground masked
    cv::Mat newBackground;// = m_recentBackgrounds.front().clone();
    m_recentBackgrounds.front().convertTo(newBackground, CV_32FC3);
    cv::Mat latest;
    m_recentFrames.front().convertTo(latest, CV_32FC3);
    latest.copyTo(newBackground, mask);

    // Foreground objects gradually become part of the background if they stop
    // changing/moving. This is only here because I suck at getting the
    // foreground perfectly and so over time I let it blend back in. If I
    // don't do this, then some pixels can get stuck as foreground forever.
    cv::addWeighted(latest, 0.05,
                    newBackground, 0.95,
                    0.0, newBackground);

    // Average in recent backgrounds
    MatList::iterator i = m_recentBackgrounds.begin();
    cv::addWeighted(newBackground, newFrameWeight, *i, frameWeight, 0.0,
                    newBackground);
    for (++i; i != m_recentBackgrounds.end(); ++i) {
        cv::addWeighted(newBackground, 1.0, *i, frameWeight, 0.0,
                        newBackground);
    }

    // Update recentBackgrounds list
    m_recentBackgrounds.push_front(newBackground);
    while (m_recentBackgrounds.size() > FRAMES_TO_TRACK) {
        m_recentBackgrounds.pop_back();
    }
}

cv::Mat
Tracker::findDeviations()
{
    cv::Mat latest;
    cv::medianBlur(m_recentFrames.front(), latest, 3);
    cv::Mat bg;
    cv::medianBlur(m_recentBackgrounds.front(), bg, 3);
    bg.convertTo(bg, CV_8UC3);

    int width = latest.size().width;
    int height = latest.size().height;

    cv::Mat result(height, width, CV_8U, cv::Scalar(0));

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            cv::Point p(x, y);
            cv::Vec3b bgvec = bg.at<cv::Vec3b>(p);
            // Is the current pixel an outlier?
            cv::Vec3b act = latest.at<cv::Vec3b>(p);
            double distance = cv::norm(act, bgvec);
            result.at<uchar>(p) = distance*255/441.673;
        }
    }

    // Smoothing?
    cv::Mat smoothed_result(height, width, CV_8U, cv::Scalar(0));
    cv::blur(result, smoothed_result, cv::Size(7, 7));

    return smoothed_result;
}

/* virtual */ TrackerResult
Tracker::track(const cv::Mat &inFrame)
{
    double scale_factor = 0.5;
    TrackerResult result;

    cv::Mat normalized;
    cv::resize(inFrame, normalized, cv::Size(), scale_factor, scale_factor,
               cv::INTER_AREA);
    cv::normalize(normalized, normalized, 0x00, 0xff, cv::NORM_MINMAX);
    m_recentFrames.push_front(normalized);
    if (m_recentFrames.size() == FRAMES_TO_TRACK) {
        initBackground();
    } else if (m_recentFrames.size() < FRAMES_TO_TRACK) {
        return result;
    }
    while (m_recentFrames.size() > FRAMES_TO_TRACK) {
        m_recentFrames.pop_back();
    }
    updateNoiseLevel();

    /* Background subtraction stuff */
    cv::Mat delta = findDeviations();
    double fgThreshold = (m_noiseLevel * 255 / 441.673) * 7;
    double bgThreshold = (m_noiseLevel * 255 / 441.673) * 3;
    cv::Mat fgMask;
    cv::Mat bgMask;
    cv::threshold(delta, fgMask, fgThreshold, 0xff, cv::THRESH_BINARY);
    cv::threshold(delta, bgMask, bgThreshold, 0xff, cv::THRESH_BINARY);
    updateBackground(~bgMask);

    fgMask = dilate(fgMask, 5);
    fgMask = dilate(fgMask, 5);
    fgMask = dilate(fgMask, 5);
    fgMask = erode(fgMask, 5);
    std::vector<cv::Rect> boundingBoxes = fillContours(fgMask);

    /* Detect massive scene changes and nuke everything if so */
    if (cv::countNonZero(fgMask) > 0.3 * normalized.size().area()) {
        std::cerr << "Major scene change.\n";
        m_noiseLevel = 441.0;
        m_recentFrames.clear();
        m_recentBackgrounds.clear();
        cv::Mat green(cv::Size(normalized.size().width/scale_factor,
                               normalized.size().height/scale_factor),
                      CV_8UC3, cv::Scalar(0, 200, 0));
        if (DEBUG_WINDOW) {
            cv::imshow(WINDOW_NAME, green);
        }
        return result;
    }

    // Some different ways of rendering the FG/BG result for debugging

    cv::Mat prettyResult;

    // Lightened foreground
    /*
    cv::Mat foregroundColor;
    cv::cvtColor(foreground, foregroundColor, CV_GRAY2BGR);
    cv::addWeighted(normalized, 1.0, foregroundColor, 0.4, 0.0, prettyResult);
    */

    // Green Outlines:
    cv::Mat contours = getContours(fgMask);
    prettyResult = normalized.clone();
    contours.copyTo(prettyResult, contours);

    // Selective color
    //prettyResult = grayout(normalized, ~fgMask);

    if (DEBUG_WINDOW) {
        cv::Size ourSize = normalized.size();
        cv::Size debugSize(ourSize.width / scale_factor,
                           ourSize.height / scale_factor);
        cv::Mat debug(debugSize, CV_8UC3, cv::Scalar(0, 0, 0));
        normalized.copyTo(debug(cv::Rect(cv::Point(0, 0), ourSize)));
        cv::putText(debug, "input", cv::Point(0, ourSize.height-5),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        prettyResult.copyTo(debug(cv::Rect(cv::Point(0, ourSize.height),
                                           ourSize)));
        cv::putText(debug, "pretty result", cv::Point(0, debugSize.height-5),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        cv::Mat deltaColor;
        cv::cvtColor(delta, deltaColor, CV_GRAY2BGR);
        deltaColor.copyTo(debug(cv::Rect(cv::Point(ourSize.width, 0),
                                         ourSize)));
        cv::putText(debug, "delta (fg)",
                    cv::Point(ourSize.width, ourSize.height-5),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));
        cv::Mat bg;
        m_recentBackgrounds.front().convertTo(bg, CV_8UC3);
        bg.copyTo(
            debug(cv::Rect(cv::Point(ourSize.width, ourSize.height), ourSize)));
        cv::putText(debug, "background",
                    cv::Point(ourSize.width, debugSize.height-5),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

        cv::imshow(WINDOW_NAME, debug);
    }

    int maxArea = 0;
    int maxI = -1;
    for (size_t i = 0; i < boundingBoxes.size(); i++) {
        if (boundingBoxes[i].area() > maxArea) {
            maxArea = boundingBoxes[i].area();
            maxI = i;
        }
    }
    // TODO: Confidence
    // TODO: Multiple boxes
    if (maxI >= 0 && maxArea > 250) {
        const cv::Rect &origBB = boundingBoxes[maxI];
        cv::Rect *outBB = new cv::Rect(origBB.tl() * (1/scale_factor),
                                       origBB.br() * (1/scale_factor));
        result.boundingBox = outBB;
        result.color = cv::Scalar(0, 0, 255);
    }
    return result;
}
