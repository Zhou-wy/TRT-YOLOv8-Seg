/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-06 10:57:10
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-15 21:31:03
 */

#include <string>

#include "../src/TrtLib/common/ilogger.hpp"
#include "../src/TrtLib/builder/trt_builder.hpp"
#include "../src/YOLOv8Seg/yolov8_seg.hpp"
#include "color_lable.hpp"


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

    ~YOLOv8SegInstance();

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
        onnx_file), m_engine_file(engine_file) {
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

YOLOv8SegInstance::~YOLOv8SegInstance() {
}

void show_result(cv::Mat &image, const YOLOv8Seg::BoxSeg &boxarray) {
    /**
     * @brief :黄色框区域 ->  安全区域
     */
    cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
    std::vector<cv::Point2i> yellowFrame{{1613, 855},
                                         {1741, 1185},
                                         {2132, 1188},
                                         {1936, 861}};

    // 在画布上绘制多边形
    std::vector<std::vector<cv::Point>> yellowFrameContours{{yellowFrame.begin(), yellowFrame.end()}};
    cv::polylines(canvas, yellowFrameContours, true, cv::Scalar(0, 255, 255), 2);
    cv::fillPoly(canvas, yellowFrameContours, cv::Scalar(0, 155, 155));

    // 叠加绘制结果到原始图像上
    cv::addWeighted(image, 1, canvas, 0.5, 0, image);

    /**
     * @brief :图像分割的结果可视化
     */
    for (auto &obj: boxarray) {
        INFO("rect: %.2f, %.2f, %.2f, %.2f, confi: %.2f, name: %s", obj.left, obj.top, obj.right, obj.bottom,
             obj.confidence, echargerlabels[obj.class_label]);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      echargercolors[obj.class_label], 3);

        auto name = echargerlabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

        // 可视化结果
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + text_width, obj.top),
                      echargercolors[obj.class_label], -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        if (obj.seg) {
            cv::Mat img_clone = image.clone();
            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
            int count = cv::countNonZero(mask == 255);
            INFO("count: %d", count);

            // 判断是否超出黄色区域
            /**
             * @todo:
             *
             *
            cv::Rect roi(rect[0].x, rect[0].y, rect[2].x - rect[0].x, rect[2].y - rect[0].y);
            cv::Mat roiMask = mask(roi);
            int count = cv::countNonZero(roiMask == 255);

            int count = 0;
            for (int row = 0; row < mask.rows; row++)
            {
                for (int col = 0; col < mask.cols; col++)
                {
                    // 获取像素值
                    uchar pixel = mask.at<uchar>(row, col);
                    // 检查像素是否为 255 且不在矩形内部
                    if (pixel == 255 )
                    {
                        if (cv::pointPolygonTest(yellowFrame, cv::Point2i(col + obj.left, row + obj.top), false) == -1)
                            count++;
                    }
                }
            }
            INFO("count: %d", count);
            */
            img_clone(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top))
                    .setTo(echargercolors[obj.class_label], mask);
            cv::addWeighted(image, 0.5, img_clone, 0.5, 1, image);
        }
    }
}

int main(int argc, char const *argv[]) {

    std::string onnx = "../workspace/model/eCharger-v8m.transd.onnx";
    std::string engine = "../workspace/model/eCharger-v8m.transd.engine";
    cv::Mat image = cv::imread("../workspace/images/over.jpg");

    iLogger::set_log_level(iLogger::LogLevel::Info);
    iLogger::set_logger_save_directory("../workspace/log/");
    std::shared_ptr<YOLOv8SegInstance> seg = std::make_shared<YOLOv8SegInstance>(onnx, engine);
    if (!seg->startup()) {
        seg.reset();
        exit(1);
    }
    YOLOv8Seg::BoxSeg boxarray;
    seg->inference(image, boxarray);
    show_result(image, boxarray);
    cv::imwrite("../workspace/images/over_result_m.jpg", image);
    return 0;
}
