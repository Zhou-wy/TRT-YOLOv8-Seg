/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-11 17:47:19
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-20 19:18:51
 */

#include "src/HttpServer/http_server.hpp"
#include "src/TrtLib/common/ilogger.hpp"
#include "src/YOLOv8Seg/yolov8_seg.hpp"
#include "src/TrtLib/builder/trt_builder.hpp"
#include "src/SqlWarpper/mysql.h"

#include <string>

#include "src/TrtLib/common/ilogger.hpp"
#include "src/TrtLib/builder/trt_builder.hpp"
#include "src/YOLOv8Seg/yolov8_seg.hpp"
#include "src/VideoPusher/rtsp2flv.hpp"

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// 用于同步的互斥锁和条件变量
std::mutex mtx;
std::condition_variable cv_producer, cv_consumer, cv_read;
// 图像队列
std::queue<cv::Mat> img_queue;

static const char *echargerlabels[] = {"Cable_1", "Cable_2"};
std::vector<cv::Scalar> echargercolors = {
    {0, 255, 0}, // Green
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
        return YOLOv8Seg::create_seg_infer(m_engine_file, task, 0, 0.65, 0.55);
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
        << R"(              \ \,\L\_\   \ \ \ \ \   \ \ \L\_\  \ \ \L\ \           \ `\`/'/\ \ \/\ \   \ \ \      \ \ \/\ \     )"
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

// 获取当前时间点
auto lastSaveTime = std::chrono::system_clock::now();

void show_result(cv::Mat &image, const YOLOv8Seg::BoxSeg &boxarray, int width, int height)
{

    // 黄色框区域 ->  安全区域
    cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
    /*
     * [0.53490324, 0.64375987, 0.71361572, 0.57785348]
     * [0.40455503, 0.40674917, 0.60466035, 0.60290504]
     * */
    std::vector<cv::Point2i> yellowFrame{{int(0.53490324 * image.cols), int(0.40455503 * image.rows)},
                                         {int(0.64375987 * image.cols), int(0.40674917 * image.rows)},
                                         {int(0.71361572 * image.cols), int(0.60466035 * image.rows)},
                                         {int(0.57785348 * image.cols), int(0.60290504 * image.rows)}};

    std::vector<std::vector<cv::Point>> yellowFrameContours{{yellowFrame.begin(), yellowFrame.end()}};
    cv::fillPoly(canvas, yellowFrameContours, cv::Scalar(0, 155, 155));
    cv::addWeighted(image, 1, canvas, 0.5, 0, image);

    bool exceedRect = false; // 判断是否超出黄色区域
    std::string det_obj = "";
    for (auto &obj : boxarray)
    {
        if (obj.seg)
        {
            cv::Mat img_clone = image.clone();
            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
            img_clone(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top))
                .setTo(exceedRect ? cv::Scalar(0, 0, 255) : echargercolors[obj.class_label], mask);
            cv::addWeighted(image, 0.5, img_clone, 0.5, 1, image);

            int all_count = cv::countNonZero(mask == 255);
            int not_in_count = 0;
            for (int row = 0; row < mask.rows; row++)
            {
                for (int col = 0; col < mask.cols; col++)
                {
                    // 获取像素值
                    uchar pixel = mask.at<uchar>(row, col);
                    // 检查像素是否为 255 且不在矩形内部
                    if (pixel == 255)
                    {
                        if (cv::pointPolygonTest(yellowFrame, cv::Point2i(col + obj.left, row + obj.top), false) == -1)
                            not_in_count++;
                    }
                }
            }

            if (float(not_in_count) / float(all_count) >= 0.4)
            {
                exceedRect = true;
                INFOD("all count: %d, not in count: %d", all_count, not_in_count);
            }
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
        det_obj.append(echargerlabels[obj.class_label]);
    }

    cv::resize(image, image, cv::Size(width, height));
    auto currentTime = std::chrono::system_clock::now();
    auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                              currentTime - lastSaveTime)
                              .count();

    // 写入数据库
    const int intervalSeconds = 5 * 60; // 延时1分钟
    if (elapsedSeconds >= intervalSeconds)
    {
        // 图片保存位置
        if (!iLogger::exists(std::string("../workspace/images/video_1")))
        {
            iLogger::mkdirs(std::string("../workspace/images/video_1"));
        }

        cv::imwrite(std::string(
                        "../workspace/images/video_1/" + iLogger::time_now() + ".jpg"),
                    image);
        // 写入数据库和关键帧
        MySQLUtil::execute("suep_echarger",
                           "insert into VideoDetectResult (Site, Date, DetObj, Result, Keyframe) values ('%s', '%s', '%s', '%s', '%s')",
                           "杨树浦路", iLogger::time_now().c_str(), det_obj.c_str(),
                           exceedRect ? "danger" : "normal",
                           std::string("../workspace/images/video_1/" + iLogger::time_now() + ".jpg").c_str());
        INFO("VideoDetectResult: %d, %s, %s, %s, %s, %s", 0, "杨树浦路",
             iLogger::time_now().c_str(), det_obj.c_str(),
             exceedRect ? "danger" : "normal",
             std::string("../workspace/images/video_1/" + iLogger::time_now() + ".jpg").c_str());
        // 更新上次保存时间
        lastSaveTime = currentTime;
    }
}

// 生产者线程函数
void SegInference(std::string in_video_url)
{
    cv::Mat src_image;
    std::string onnx = "../workspace/model/eCharger-v8x.transd.onnx";
    std::string engine = "../workspace/model/eCharger-v8x.transd.engine";
    std::shared_ptr<YOLOv8SegInstance> seg = std::make_shared<YOLOv8SegInstance>(onnx, engine);
    if (!seg->startup())
    {
        seg.reset();
        exit(1);
    }

    srand((unsigned)time(NULL));
    INFO("opencv version: %s", CV_VERSION);

    cv::VideoCapture cap = cv::VideoCapture(in_video_url);
    if (!cap.isOpened())
    {
        INFOE("Error opening video stream or file");
        return;
    }
    YOLOv8Seg::BoxSeg boxarray;
    while (true)
    {
        try
        {
            if (!cap.isOpened())
            {
                INFOE("can not read image from video: %s", in_video_url.c_str());
                // 如果无法读取到图像，等待1min后再次尝试
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                cap = cv::VideoCapture(in_video_url);
            }
            cap.read(src_image);
            cv::Mat dst_image;
            cv::resize(src_image, dst_image, cv::Size(640, int((640.0 / src_image.cols) * src_image.rows)));
            seg->inference(dst_image, boxarray);
            show_result(dst_image, boxarray, src_image.cols, src_image.rows);
            // 加锁队列
            std::unique_lock<std::mutex> lock(mtx);
            // 队列满，等待消费者消费
            cv_producer.wait(lock, []()
                             {
                        bool is_full = img_queue.size() < 30;
                        if (!is_full) {
                            INFO("Producer is waiting...");
                        }
                        return is_full; });
            // 图像加入队列
            img_queue.push(dst_image);
            // 通知消费者
            cv_consumer.notify_all();
        }
        catch (const std::exception &ex)
        {
            INFOE("Error occurred in producer_thread: %s", ex.what());
        }
    }
}

// 消费者线程函数
void video2flv(double width, double height, int fps, int bitrate, std::string codec_profile, std::string out_url)
{

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
#endif
    avformat_network_init();

    const char *output = out_url.c_str();
    // std::vector<uint8_t> imgbuf(height * width * 3 + 16);
    // cv::Mat image(height, width, CV_8UC3, imgbuf.data(), width * 3);

    AVFormatContext *ofmt_ctx = nullptr;
    const AVCodec *out_codec = nullptr;
    AVStream *out_stream = nullptr;
    AVCodecContext *out_codec_ctx = nullptr;

    initialize_avformat_context(ofmt_ctx, "flv");
    initialize_io_context(ofmt_ctx, output);

    out_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    out_stream = avformat_new_stream(ofmt_ctx, out_codec);
    out_codec_ctx = avcodec_alloc_context3(out_codec);

    set_codec_params(ofmt_ctx, out_codec_ctx, width, height, fps, bitrate);
    initialize_codec_stream(out_stream, out_codec_ctx, out_codec, codec_profile);

    out_stream->codecpar->extradata = out_codec_ctx->extradata;
    out_stream->codecpar->extradata_size = out_codec_ctx->extradata_size;

    av_dump_format(ofmt_ctx, 0, output, 1);

    auto *swsctx = initialize_sample_scaler(out_codec_ctx, width, height);
    auto *frame = allocate_frame_buffer(out_codec_ctx, width, height);

    int cur_size;
    uint8_t *cur_ptr;

    int ret = avformat_write_header(ofmt_ctx, nullptr);
    if (ret < 0)
    {
        INFOE("Could not write header!");
        exit(1);
    }

    //    bool end_of_stream = false;
    INFO("begin to flv url: %s", output);
    while (true)
    {
        try
        {
            // 加锁队列
            std::unique_lock<std::mutex> lock(mtx);
            INFOD("ffmpeg: the size of image queue : %d ", img_queue.size());

            // 队列空，等待生产者生产
            cv_consumer.wait(lock, []()
                             { return !img_queue.empty(); });

            // 取出队首图像
            cv::Mat image = img_queue.front();
            img_queue.pop();

            // 解锁队列
            lock.unlock();
            cv_producer.notify_one();

            // 消费图像consumeImage(image);
            cv::resize(image, image, cv::Size(width, height));
            const int stride[] = {static_cast<int>(image.step[0])};
            sws_scale(swsctx, &image.data, stride, 0, image.rows, frame->data, frame->linesize);
            frame->pts += av_rescale_q(1, out_codec_ctx->time_base, out_stream->time_base);
            write_frame(out_codec_ctx, ofmt_ctx, frame);
        }
        catch (const std::exception &ex)
        {
            INFOE("Error occurred in consumer_thread: %s", ex.what());
        }
    }
    av_write_trailer(ofmt_ctx);

    av_frame_free(&frame);
    avcodec_close(out_codec_ctx);
    avio_close(ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
}

int main(int argc, char const *argv[])
{

    // 数据库初始化
    mysql::MySQLMgr::GetInstance()->add("suep_echarger", "127.0.0.1", 3306, "root", "12345678", "eCharger");

    // 推流参数
    std::string in_url = "https://hzhls05.ys7.com:7994/v3/openlive/G54186723_3_2.m3u8?expire=1689928610&id=602177108236509184&t=800ecf6eec889973e9f34ed1dc80879da9f0b1a79866334038c63f062e2a6ca1&ev=100&u=7f604abe45c24f7794eecc1e80adfe98";
    int fps = 25, width = 1280, height = 720, bitrate = 300000;
    std::string h264profile = "high444"; //(baseline | high | high10 | high422 | high444 | main) (default: high444)"
    std::string out_url = "rtmp://192.168.0.113:1935/myapp/mystream";

    // 日志设置
    iLogger::set_log_level(iLogger::LogLevel::Info);
    iLogger::set_logger_save_directory("../workspace/log");

    // 创建生产者和消费者线程
    std::thread producer(SegInference, in_url);
    std::thread consumer(video2flv, width, height, fps, bitrate, h264profile, out_url);

    // 等待线程结束
    if (producer.joinable())
        producer.join();
    if (consumer.joinable())
        consumer.join();
    return 0;
}
/**
 * Input #0, flv, from 'rtmp://192.168.0.113:1935/myapp/mystream':/0
  Metadata:
    |RtmpSampleAccess: true
    Server          : NGINX HTTP-FLV (https://github.com/winshining/nginx-http-flv-module)
    displayWidth    : 1280
    displayHeight   : 720
    fps             : 0
    profile         :
    level           :
    Duration: 00:00:00.00, start: 0.000000, bitrate: N/A
  Stream #0:0: Video: h264 (Main), yuv420p(progressive), 1280x720, 9000 kb/s, 25 fps, 25 tbr, 1k tbn
  11.84 M-V:  0.874 fd=  12 aq=    0KB vq=    0KB sq=    0B f=0/0
 * */

/**
 *Input #0, hls, from 'https://xy3hls02.ys7.com:7990/v3/openlive/G54186723_3_2.m3u8?expire=1689754817&id=601448168831328256&t=ca872a072ed3f0035dd20df4a6997f378acf66bd6b2ecf1cb93a592fd7d4ec82&ev=100&u=d1b08689e3cf4bc383c99fa88837c1ab':
  Duration: N/A, start: 606.000000, bitrate: N/A
  Program 0
    Metadata:
      variant_bitrate : 0
  Stream #0:0: Video: h264 (Main) ([27][0][0][0] / 0x001B), yuvj420p(pc, bt709), 1280x720, 25 fps, 25 tbr, 90k tbn
    Metadata:
      variant_bitrate : 0
[https @ 0x147881c00] Opening 'https://xy3hls02.ys7.com:7990/openlivedata/G54186723_3_2/daec040f0215457a8a5a6addc1673787-306.ts?Usr=11b8a93e1cc546aea6a675e1e5b0cf21' for reading
  Duration: N/A, start: 606.000000, bitrate: N/A
  Program 0
    Metadata:
      variant_bitrate : 0
  Stream #0:0: Video: h264 (Main) ([27][0][0][0] / 0x001B), yuvj420p(pc, bt709), 1280x720, 25 fps, 25 tbr, 90k tbn
    Metadata:
      variant_bitrate : 0
 * */
