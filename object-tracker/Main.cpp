#include <iomanip>
#include <iostream>
#include <sstream>

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

//#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tracker.h"

// Set to non-zero to enable recording videos when motion is detected
#define RECORD_MOTION_VIDEOS 0
// Make sure this directory exists if you enable recording
#define RECORDINGS_DIR "caps/"
#define WINDOW_NAME "Object Tracker"

cv::Mat
decorate(const cv::Mat &inFrame, const TrackerResult &result)
{
    cv::Mat outFrame = inFrame.clone();
    if (result.boundingBox) {
        // TEMPORARY: Disable drawing bounding boxes so that I get some
        // unretouched videos for testing purposes.
        //cv::rectangle(outFrame, result.boundingBox->tl(),
        //              result.boundingBox->br(), result.color, 1, 8, 0);
    }
    return outFrame;
}

std::string
makefilename(const std::string &prefix, const std::string &ext)
{
    std::ostringstream result;
    for (int i = 1; i < 10000; i++) {
        struct stat buf;
        result << prefix << "out-" << std::setfill('0')<<std::setw(4)<<i<< ext;
        int s = stat(result.str().c_str(), &buf);
        if (s != 0) {
            if (errno == ENOENT) {
                break;
            } else {
                std::cerr << "Unexpected error testing file: " << result << "\n";
            }
        }
        result.str("");
    }
    return result.str();
}

int
watchStream(const std::string &path)
{
    std::cerr << "Opening URL: " << path << std::endl;
    std::cerr << (RECORD_MOTION_VIDEOS ? "WILL" : "Will NOT")
              << " record motion capture videos." << std::endl;

    cv::Mat inFrame;
    cv::Mat outFrame;
    std::cerr << "Opening stream... (patience is a virtue)\n";
    cv::VideoCapture capture(path);
    cv::namedWindow(WINDOW_NAME);

    // Apologies for the mess that follows. :)
    bool done = false;
    bool paused = false;
    cv::VideoWriter *writer = NULL;
    cv::VideoWriter *motionCapture = NULL;
    int frames = 0;
    int frame_delay = 10;
    int t1 = time(NULL);
    int t2;
    int stopMotionCapture = 0;
    double fps;
    Tracker tracker;
    while (!done && capture.isOpened()) {
        if (!paused) {
            if (!capture.grab()) {
                std::cerr << "no grab\n";
                break;
            }
            capture.retrieve(inFrame);
            if (inFrame.empty()) {
                std::cerr << "empty frame\n";
                continue;
            }
            frames += 1;
            t2 = time(NULL);
            if (t2 - t1 > 5) {
                int elapsed = t2 - t1;
                fps = ((double) frames) / elapsed;
                frames = 0;
                t1 = time(NULL);
                std::cerr << "Apparent FPS: " << fps << "                \r";
            }
            if (writer) {
                writer->write(inFrame);
            }
            outFrame = inFrame;
            TrackerResult result = tracker.track(inFrame);
            if (result.boundingBox) {
                if (RECORD_MOTION_VIDEOS) {
                    std::string path = makefilename(RECORDINGS_DIR, ".mjpg");
                    motionCapture = new cv::VideoWriter(
                               path, CV_FOURCC('M','J','P','G'), fps,
                               outFrame.size());
                    std::cerr << "Started motion capture to "
                              << path << "...\n";
                }
                stopMotionCapture = time(NULL) + 5;
            }
            outFrame = decorate(outFrame, result);
            if (motionCapture && time(NULL) >= stopMotionCapture) {
                delete motionCapture;
                motionCapture = NULL;
                std::cerr << "Stopped motion capture.\n";
            }
            if (motionCapture) {
                motionCapture->write(outFrame);
            }
        }
        cv::imshow(WINDOW_NAME, outFrame);
        switch (cv::waitKey(frame_delay)) {
            case 27:
                done = true;
                break;
            case ' ':
                paused = !paused;
                if (paused) {
                    std::cerr << "Paused.\n";
                } else {
                    frames = 0;
                    t1 = time(NULL);
                    std::cerr << "Unpaused.\n";
                }
                break;
            case '+':
            case '=':
                frame_delay -= 10;
                if (frame_delay < 10) {
                    frame_delay = 10;
                }
                std::cerr << "New inter-frame delay: " << frame_delay << "ms\n";
                break;
            case '-':
                frame_delay += 10;
                std::cerr << "New inter-frame delay: " << frame_delay << "ms\n";
                break;
            case 'r': {
                if (writer == NULL) {
                    std::string path = makefilename("", ".mjpg");
                    writer = new cv::VideoWriter(
                               path, CV_FOURCC('M','J','P','G'), fps,
                               outFrame.size());
                    std::cerr << "Started recording to " << path << "...\n";
                } else {
                    delete writer;
                    writer = NULL;
                    std::cerr << "Recording stopped.\n";
                }
            }
            break;
        }
    }

    cv::destroyWindow(WINDOW_NAME);

    if (writer) {
        delete writer;
        writer = NULL;
        std::cerr << "Recording stopped.\n";
    }
    if (motionCapture) {
        delete motionCapture;
        motionCapture = NULL;
        std::cerr << "Motion capture stopped.\n";
    }

    return !done;
}

int
main(int argc, char *argv[])
{
    std::cerr << "Compiled with OpenCV " << CV_VERSION << std::endl;

    if (argc == 1) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " [url or path]\n";
        return 0;
    }

    return watchStream(argv[1]);
}
