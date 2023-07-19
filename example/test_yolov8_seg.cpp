/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-06 10:57:10
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-16 17:13:37
 */

#include <string>

#include "../src/TrtLib/common/ilogger.hpp"
#include "../src/TrtLib/builder/trt_builder.hpp"
#include "../src/YOLOv8Seg/yolov8_seg.hpp"
#include "color_lable.hpp"

class YOLOv8SegInstance
{
private:
    std::string m_engine_file;
    std::string m_onnx_file;
    std::shared_ptr<YOLOv8Seg::SegInfer> SegIns;

    std::shared_ptr<YOLOv8Seg::SegInfer> get_infer(YOLOv8Seg::Task task)
    {
        if (!iLogger::exists(m_engine_file))
        {
            TRT::compile(
                TRT::Mode::FP32,
                10,
                m_onnx_file,
                m_engine_file);
        }
        else
        {
            INFOW("%s has been created!", m_engine_file.c_str());
        }
        return YOLOv8Seg::create_seg_infer(m_engine_file, task, 0);
    }

public:
    YOLOv8SegInstance(const std::string &onnx_file, const std::string &engine_file);

    ~YOLOv8SegInstance();

    bool startup()
    {
        SegIns = get_infer(YOLOv8Seg::Task::seg);
        return SegIns != nullptr;
    }

    bool inference(const cv::Mat &image_input, YOLOv8Seg::BoxSeg &boxarray)
    {

        if (SegIns == nullptr)
        {
            INFOE("Not Initialize.");
            return false;
        }

        if (image_input.empty())
        {
            INFOE("Image is empty.");
            return false;
        }
        boxarray = SegIns->commit(image_input).get();
        return true;
    }
};

YOLOv8SegInstance::YOLOv8SegInstance(const std::string &onnx_file, const std::string &engine_file) : m_onnx_file(
                                                                                                         onnx_file),
                                                                                                     m_engine_file(engine_file)
{
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

YOLOv8SegInstance::~YOLOv8SegInstance()
{
}


void show_result(cv::Mat &image, const YOLOv8Seg::BoxSeg &boxarray, int width, int height) {

    //黄色框区域 ->  安全区域
    cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
    std::vector<cv::Point2i> yellowFrame{{int(0.5334 * image.cols), int(0.4353 * image.rows)},
                                         {int(0.5757 * image.cols), int(0.6034 * image.rows)},
                                         {int(0.7050 * image.cols), int(0.6049 * image.rows)},
                                         {int(0.6402 * image.cols), int(0.4384 * image.rows)}};
    // 在画布上绘制多边形
    std::vector<std::vector<cv::Point>> yellowFrameContours{{yellowFrame.begin(), yellowFrame.end()}};
    cv::fillPoly(canvas, yellowFrameContours, cv::Scalar(0, 155, 155));

    // 叠加绘制结果到原始图像上
    cv::addWeighted(image, 1, canvas, 0.5, 0, image);

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

            if (float(not_in_count) / float(all_count) >= 0.3) {
                exceedRect = true;
                INFOD("all count: %d, not in count: %d", all_count, not_in_count);
            }

            img_clone(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top))
                    .setTo(exceedRect ? cv::Scalar(0, 0, 255) : echargercolors[obj.class_label], mask);
            cv::addWeighted(image, 0.6, img_clone, 0.4, 1, image);
        }

        INFOD("rect: %.2f, %.2f, %.2f, %.2f, confi: %.2f, name: %s", obj.left, obj.top, obj.right, obj.bottom,
              obj.confidence, echargerlabels[obj.class_label]);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      exceedRect ? cv::Scalar(0, 0, 255) : echargercolors[obj.class_label], 1);

        auto caption = cv::format("%s %.2f", echargerlabels[obj.class_label], obj.confidence);
        int text_width = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 10;

        // 可视化结果
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 20), cv::Point(obj.left + text_width, obj.top),
                      exceedRect ? cv::Scalar(0, 0, 255) : echargercolors[obj.class_label], -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 0.5, cv::Scalar::all(0), 1, 8);
        cv::resize(image, image, cv::Size(width, height));


//        auto currentTime = std::chrono::system_clock::now();
//        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
//                currentTime - lastSaveTime).count();

        // 写入数据库
        const int intervalSeconds = 5 * 60; // 延时5分钟
        if (1) {
            //图片保存位置
            if (!iLogger::exists(std::string("../workspace/images/video_1"))) {
                iLogger::mkdirs(std::string("../workspace/images/video_1"));
            }

            cv::imwrite(std::string(
                    "../workspace/images/video_1/" + iLogger::time_now() + ".jpg"), image);
            // 写入数据库和关键帧
//            MySQLUtil::execute("suep_echarger",
//                               "insert into VideoDetectResult (ID, Site, Date, DetObj, Result, Keyframe) values ('%d', '%s', '%s', '%s', '%s', '%s')",
//                               save_conut, "杨树浦路", iLogger::time_now().c_str(), echargerlabels[obj.class_label],
//                               exceedRect ? "danger" : "normal",
//                               std::string("../workspace/images/video_1/" + iLogger::time_now() + ".jpg").c_str());
            INFO("VideoDetectResult: %d, %s, %s, %s, %s, %s", 1, "杨树浦路",
                 iLogger::time_now().c_str(), echargerlabels[obj.class_label],
                 exceedRect ? "danger" : "normal",
                 std::string("../workspace/images/video_1/" + iLogger::time_now() + ".jpg").c_str());

//            // 更新上次保存时间
//            lastSaveTime = currentTime;
//            save_conut++;
        }
    }
}


int main(int argc, char const *argv[])
{

    std::string onnx = "../workspace/model/eCharger-v8m.transd.onnx";
    std::string engine = "../workspace/model/eCharger-v8m.transd.engine";
    cv::Mat image = cv::imread("../workspace/images/over.jpg");

    iLogger::set_log_level(iLogger::LogLevel::Info);
    iLogger::set_logger_save_directory("../workspace/log/");
    std::shared_ptr<YOLOv8SegInstance> seg = std::make_shared<YOLOv8SegInstance>(onnx, engine);
    if (!seg->startup())
    {
        seg.reset();
        exit(1);
    }
    YOLOv8Seg::BoxSeg boxarray;
    cv::Mat resize_img;
    cv::resize(image, resize_img, cv::Size(640, int((640.0 / image.cols) * image.rows)));
    seg->inference(resize_img, boxarray);
    show_result(resize_img, boxarray, image.cols, image.rows);
//    cv::resize(resize_img, image, cv::Size(image.cols, image.rows));

    cv::imwrite("../workspace/images/over_result_m.jpg", image);
    return 0;
}
