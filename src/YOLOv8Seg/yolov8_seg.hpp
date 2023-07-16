#ifndef __YOLOV8_SEG_HPP
#define __YOLOV8_SEG_HPP

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <future>

#include "segment.hpp"

namespace YOLOv8Seg
{
    typedef std::vector<Box> BoxSeg;
    enum class Task : int
    {
        det = 0, // detect task
        seg = 1, // segment task
        clf = 2, // classify task
        pos = 3  // posture task
    };

    enum class NMSMethod : int
    {
        CPU = 0,    // General, for estimate mAP
        FastGPU = 1 // Fast NMS with a small loss of accuracy in corner cases
    };

    class SegInfer
    {
    public:
        virtual std::shared_future<BoxSeg> commit(const cv::Mat &image) = 0;
        virtual std::vector<std::shared_future<BoxSeg>> commits(const std::vector<cv::Mat> &images) = 0;
    };

    std::shared_ptr<SegInfer> create_seg_infer(
        const std::string &engine_file, Task task = Task::seg, int gpuid = 0,
        float confidence_threshold = 0.25f, float nms_threshold = 0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false);
};


#endif