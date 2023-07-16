/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-11 17:47:19
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-13 16:26:23
 */

#include "src/HttpServer/http_server.hpp"
#include "src/TrtLib/common/ilogger.hpp"
#include "src/YOLOv8Seg/yolov8_seg.hpp"
#include "src/TrtLib/builder/trt_builder.hpp"



static const char *echargerlabels[] = {"ChargeCable_1", "ChargeCable_2"};
std::vector<cv::Scalar> echargercolors = {
    {0, 0, 255}, // Red
    {255, 0, 0}, // Bule
};

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
    ~YOLOv8SegInstance(){};

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

YOLOv8SegInstance::YOLOv8SegInstance(const std::string &onnx_file, const std::string &engine_file) : m_onnx_file(onnx_file), m_engine_file(engine_file)
{
    std::cout << "                       " << std::endl;
    std::cout << "               ____        __  __      ____       ____                __    __  _____       __         _____        " << std::endl;
    std::cout << "              /\\  _`\\     /\\ \\/\\ \\    /\\  _`\\    /\\  _`\\             /\\ \\  /\\ \\/\\  __`\\    /\\ \\       /\\  __`\\      " << std::endl;
    std::cout << "              \\ \\,\\L\\_\\   \\ \\ \\ \\ \\   \\ \\ \\L\\_\\  \\ \\ \\L\\ \\           \\ `\\`\\\\/'/\\ \\ \\/\\ \\   \\ \\ \\      \\ \\ \\/\\ \\     " << std::endl;
    std::cout << "               \\/_\\__ \\    \\ \\ \\ \\ \\   \\ \\  _\\L   \\ \\ ,__/            `\\ `\\ /'  \\ \\ \\ \\ \\   \\ \\ \\  __  \\ \\ \\ \\ \\    " << std::endl;
    std::cout << "                 /\\ \\L\\ \\   \\ \\ \\_\\ \\   \\ \\ \\L\\ \\  \\ \\ \\/               `\\ \\ \\   \\ \\ \\_\\ \\   \\ \\ \\L\\ \\  \\ \\ \\_\\ \\   " << std::endl;
    std::cout << "                 \\ `\\____\\   \\ \\_____\\   \\ \\____/   \\ \\_\\                 \\ \\_\\   \\ \\_____\\   \\ \\____/   \\ \\_____\\  " << std::endl;
    std::cout << "                  \\/_____/    \\/_____/    \\/___/     \\/_/                  \\/_/    \\/_____/    \\/___/     \\/_____/  " << std::endl;
    std::cout << "                       " << std::endl;
}



class HttpController : public Controller
{
    SetupController(HttpController);

public:
    bool startup();

public:
    DefRequestMapping(HelloHttpServer);
    DefRequestMapping(SegmentTest);

private:
    std::shared_ptr<YOLOv8SegInstance> Seg_Instance_;
};

Json::Value HttpController::HelloHttpServer(const Json::Value &param)
{

    Json::Value data;
    data["alpha"] = 199897;
    data["beta"] = "weiyi";
    data["name"] = "周威仪";
    return success(data);
}

Json::Value HttpController::SegmentTest(const Json::Value &param)
{
    cv::Mat image = cv::imread("../workspace/images/echarger.jpg");

    YOLOv8Seg::BoxSeg boxarray;
    Json::Value json(Json::arrayValue);

    int width = image.cols;
    int height = image.rows;

    if (!this->Seg_Instance_->inference(image, boxarray))
        return failure("Server error1");

    for (auto &box : boxarray)
    {
        Json::Value det_result(Json::objectValue);
        det_result["left"] = box.left / width;
        det_result["top"] = box.top / height;
        det_result["right"] = box.right / width;
        det_result["bottom"] = box.bottom / height;
        det_result["confidence"] = box.confidence;
        det_result["class_label"] = box.class_label;
        det_result["class_name"] = echargerlabels[box.class_label];
        json.append(det_result);
    }

    Json::Value yellow_rect(Json::objectValue);
    yellow_rect["class_label"] = "YellowRect";
    yellow_rect["left_top_x"] = 1613 / width;
    yellow_rect["left_top_y"] = 855 / height;
    yellow_rect["right_top_x"] = 1741 / width;
    yellow_rect["right_top_y"] = 1185 / height;
    yellow_rect["left_bottom_x"] = 2132 / width;
    yellow_rect["left_bottom_y"] = 1188 / height;
    yellow_rect["right_bottom_x"] = 1613 / width;
    yellow_rect["right_bottom_x"] = 855 / height;
    json.append(yellow_rect);

    return success(json);
}

bool HttpController::startup()
{
    std::string onnx = "../workspace/model/echarger-v8x.transd.onnx";
    std::string engine = "../workspace/model/echarger-v8x.transd.engine";
    Seg_Instance_ = std::make_shared<YOLOv8SegInstance>(onnx, engine);
    if (!Seg_Instance_->startup())
    {
        Seg_Instance_.reset();
    }
    return Seg_Instance_ != nullptr;
}


int test_http(int port = 8090)
{

    INFO("Create controller");
    auto logical_controller = std::make_shared<HttpController>();
    if (!logical_controller->startup())
    {
        INFOE("Startup controller failed.");
        return -1;
    }

    std::string address = iLogger::format("192.168.0.113:%d", port);
    INFO("Create http server to: %s", address.c_str());

    auto server = createHttpServer(address, 32);
    if (!server)
        return -1;
    server->verbose();

    INFO("Add controller");
    server->add_controller("/api", logical_controller);
    // server->add_controller("/", create_redirect_access_controller("./web"));
    // server->add_controller("/static", create_file_access_controller("./"));
    INFO("Access url: http://%s", address.c_str());

    INFO(
        "访问如下地址即可看到效果:\n"
        "1. http://%s/api/HelloHttpServer             测试http server 是否正常启动\n"
        "2. http://%s/api/SegmentTest                 测试预测图片结果是否正常\n",
        address.c_str(), address.c_str());

    INFO("按下Ctrl + C结束程序");
    return iLogger::while_loop();
}

int main(int argc, char const *argv[])
{
    test_http();
    return 0;
}
