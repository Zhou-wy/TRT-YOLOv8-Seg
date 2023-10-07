/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-06 10:57:10
 * @LastEditors: zwy
 * @LastEditTime: 2023-10-07 14:25:11
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
        INFO("===================== test YOLOv8 int8 ==================================");
        auto int8process = [=](int current, int count, const std::vector<std::string> &files, std::shared_ptr<TRT::Tensor> &tensor)
        {
            INFO("Int8 %d / %d", current, count);

            for (int i = 0; i < files.size(); ++i)
            {
                auto image = cv::imread(files[i]);
                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                cv::resize(image, image, cv::Size(tensor->size(3), tensor->size(2)));
                image.convertTo(image, CV_32F, 1 / 255.0f);
                tensor->set_mat(i, image);
            }
        };

        if (!iLogger::exists(m_engine_file))
        {
            TRT::compile(
                TRT::Mode::INT8,
                10,
                m_onnx_file,
                m_engine_file,
                {},
                int8process,
                "/home/zwy/PyWorkspace/eCharger/TRT_YOLOv8_Server/workspace/media",
                "/home/zwy/PyWorkspace/eCharger/TRT_YOLOv8_Server/workspace/media/calibration.cache");
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
        SegIns = get_infer(YOLOv8Seg::Task::det);
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

void show_result(cv::Mat &image, const YOLOv8Seg::BoxSeg &boxarray, int width, int height)
{

    for (auto &obj : boxarray)
    {
        if (obj.seg)
        {
            cv::Mat img_clone = image.clone();
            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);


            img_clone(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top))
                .setTo(coco_colors[obj.class_label], mask);
            cv::addWeighted(image, 0.6, img_clone, 0.4, 1, image);
        }

        INFOD("rect: %.2f, %.2f, %.2f, %.2f, confi: %.2f, name: %s", obj.left, obj.top, obj.right, obj.bottom,
              obj.confidence, coco_labels[obj.class_label]);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      coco_colors[obj.class_label], 1);

        auto caption = cv::format("%s %.2f", coco_labels[obj.class_label], obj.confidence);
        int text_width = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 10;

        // 可视化结果
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 20), cv::Point(obj.left + text_width, obj.top),
                      coco_colors[obj.class_label], -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 0.5, cv::Scalar::all(0), 1, 8);
    }
}

int main(int argc, char const *argv[])
{

    std::string onnx = "../workspace/model/yolov8n.transd.onnx";
    std::string engine = "../workspace/model/yolov8n.transd.int8.engine";
    cv::Mat image = cv::imread("../workspace/images/car.jpg");

    iLogger::set_log_level(iLogger::LogLevel::Info);
    iLogger::set_logger_save_directory("../workspace/log/");
    std::shared_ptr<YOLOv8SegInstance> seg = std::make_shared<YOLOv8SegInstance>(onnx, engine);
    if (!seg->startup())
    {
        seg.reset();
        exit(1);
    }
    YOLOv8Seg::BoxSeg boxarray;
    seg->inference(image, boxarray);

    show_result(image, boxarray, image.cols, image.rows);
    for (auto box : boxarray)
    {
        std::cout <<"["<< box.left << "," << box.top << "," << box.right << "," << box.bottom << "]" << box.class_label << " " << box.confidence << std::endl;
    }

    cv::imwrite("../workspace/images/car_result_det_INT8.jpg", image);
    return 0;
}
