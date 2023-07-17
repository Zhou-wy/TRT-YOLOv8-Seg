//
// Created by zhouwy on 2023/7/17.
//


// YOLOv8 segment include
#include <string>

#include "TrtLib/common/ilogger.hpp"
#include "TrtLib/builder/trt_builder.hpp"
#include "YOLOv8Seg/yolov8_seg.hpp"

/**
 * @brief: YOLOv8 segmentation instance
 * */
class YOLOv8SegInstance {
private:
    std::string m_engine_file;
    std::string m_onnx_file;
    std::shared_ptr<YOLOv8Seg::SegInfer> SegIns;

    std::shared_ptr<YOLOv8Seg::SegInfer> get_infer(YOLOv8Seg::Task task) {
        if (!iLogger::exists(m_engine_file)) {
            TRT::compile(
                    TRT::Mode::FP32,
                    10,
                    m_onnx_file,
                    m_engine_file);
        } else {
            INFOW("%s has been created!", m_engine_file.c_str());
        }
        return YOLOv8Seg::create_seg_infer(m_engine_file, task, 0);
    }

public:
    YOLOv8SegInstance(const std::string &onnx_file, const std::string &engine_file);

    ~YOLOv8SegInstance() = default;

    bool startup() {
        SegIns = get_infer(YOLOv8Seg::Task::seg);
        return SegIns != nullptr;
    }

    bool inference(const cv::Mat &image_input, YOLOv8Seg::BoxSeg &boxarray) {

        if (SegIns == nullptr) {
            INFOE("Not Initialize.");
            return false;
        }

        if (image_input.empty()) {
            INFOE("Image is empty.");
            return false;
        }
        boxarray = SegIns->commit(image_input).get();
        return true;
    }
};

YOLOv8SegInstance::YOLOv8SegInstance(const std::string &onnx_file, const std::string &engine_file) : m_onnx_file(
        onnx_file),
                                                                                                     m_engine_file(
                                                                                                             engine_file) {
    std::cout << "                       " << std::endl;
    std::cout
            << R"(               ____        __  __      ____       ____                __    __  _____       __         _____        )"
            << std::endl;
    std::cout
            << R"(              /\  _`\     /\ \/\ \    /\  _`\    /\  _`\             /\ \  /\ \/\  __`\    /\ \       /\  __`\      )"
            << std::endl;
    std::cout
            << R"(              \ \,\L\_\   \ \ \ \ \   \ \ \L\_\  \ \ \L\ \           \ `\`\\/'/\ \ \/\ \   \ \ \      \ \ \/\ \     )"
            << std::endl;
    std::cout
            << R"(               \/_\__ \    \ \ \ \ \   \ \  _\L   \ \ ,__/            `\ `\ /'  \ \ \ \ \   \ \ \  __  \ \ \ \ \    )"
            << std::endl;
    std::cout
            << R"(                 /\ \L\ \   \ \ \_\ \   \ \ \L\ \  \ \ \/               `\ \ \   \ \ \_\ \   \ \ \L\ \  \ \ \_\ \   )"
            << std::endl;
    std::cout
            << R"(                 \ `\____\   \ \_____\   \ \____/   \ \_\                 \ \_\   \ \_____\   \ \____/   \ \_____\  )"
            << std::endl;
    std::cout
            << R"(                  \/_____/    \/_____/    \/___/     \/_/                  \/_/    \/_____/    \/___/     \/_____/  )"
            << std::endl;
    std::cout << "                       " << std::endl;
}


void show_result(cv::Mat &image, const YOLOv8Seg::BoxSeg &boxarray) {
    static const char *echarger_labels[] = {"Cable-1", "Cable-2"};
    std::vector<cv::Scalar> echarger_colors = {
            {160, 82, 45}, // Sienna
            {220, 20, 60}, // Crimson
    };
    cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
    // 黄色区域
    std::vector<cv::Point2i> yellowFrame{{int(0.5334 * image.cols), int(0.4353 * image.rows)},
                                         {int(0.5757 * image.cols), int(0.6034 * image.rows)},
                                         {int(0.7050 * image.cols), int(0.6049 * image.rows)},
                                         {int(0.6402 * image.cols), int(0.4384 * image.rows)}};

    // 图像分割的结果可视化
    bool exceedRect = false; // 判断是否超出黄色区域

    for (auto &obj: boxarray) {
        if (obj.seg) {
            cv::Mat img_clone = image.clone();
            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);

            int all_count = cv::countNonZero(mask == 255);
            int not_in_count = 0;
            for (int row = 0; row < mask.rows; row++) {
                for (int col = 0; col < mask.cols; col++) {
                    // 获取像素值
                    uchar pixel = mask.at<uchar>(row, col);
                    // 检查像素是否为 255 且不在矩形内部
                    if (pixel == 255) {
                        if (cv::pointPolygonTest(yellowFrame, cv::Point2i(col + obj.left, row + obj.top), false) == -1)
                            not_in_count++;
                    }
                }
            }

            if (float(not_in_count / all_count) >= 0.1) {
                exceedRect = true;
            }
            INFO("all count: %d, not in count: %d", all_count, not_in_count);
            img_clone(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top))
                    .setTo(echarger_colors[obj.class_label], mask);
            cv::addWeighted(image, 0.5, img_clone, 0.5, 1, image);
        }

        INFO("rect: %.2f, %.2f, %.2f, %.2f, confi: %.2f, name: %s", obj.left, obj.top, obj.right, obj.bottom,
             obj.confidence, echarger_labels[obj.class_label]);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      echarger_colors[obj.class_label], 1);

        auto name = echarger_labels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int text_width = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 10;

        // 可视化结果
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 20), cv::Point(obj.left + text_width, obj.top),
                      echarger_colors[obj.class_label], -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 0.5, cv::Scalar::all(0), 1, 8);
    }
    INFO("exceedRect: %d", exceedRect);
    cv::Scalar RectColor = exceedRect ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 155, 155);

    // 在画布上绘制多边形
    std::vector<std::vector<cv::Point>> yellowFrameContours{{yellowFrame.begin(), yellowFrame.end()}};
    cv::polylines(canvas, yellowFrameContours, true, RectColor, 2);
    cv::fillPoly(canvas, yellowFrameContours, RectColor);

    // 叠加绘制结果到原始图像上
    cv::addWeighted(image, 1, canvas, 0.5, 0, image);
}


