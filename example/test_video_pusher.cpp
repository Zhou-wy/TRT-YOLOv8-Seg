/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-13 16:28:02
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-15 20:49:46
 */
#include <string>

#include "../src/TrtLib/common/ilogger.hpp"
#include "../src/TrtLib/builder/trt_builder.hpp"
#include "../src/YOLOv8Seg/yolov8_seg.hpp"
#include "color_lable.hpp"
#include "../src/VideoPusher/rtsp2flv.hpp"
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// 用于同步的互斥锁和条件变量
std::mutex mtx;
std::condition_variable cv_producer, cv_consumer;
// 图像队列
std::queue<cv::Mat> img_queue;

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

void show_result(cv::Mat &image, const YOLOv8Seg::BoxSeg &boxarray)
{
    /**
     * @brief :黄色框区域 ->  安全区域
     */
    // cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
    // std::vector<cv::Point2i> yellowFrame{{1613, 855}, {1741, 1185}, {2132, 1188}, {1936, 861}};

    // // 在画布上绘制多边形
    // std::vector<std::vector<cv::Point>> yellowFrameContours{{yellowFrame.begin(), yellowFrame.end()}};
    // cv::polylines(canvas, yellowFrameContours, true, cv::Scalar(0, 255, 255), 2);
    // cv::fillPoly(canvas, yellowFrameContours, cv::Scalar(0, 155, 155));

    // // 叠加绘制结果到原始图像上
    // cv::addWeighted(image, 1, canvas, 0.5, 0, image);

    /**
     * @brief :图像分割的结果可视化
     */
    for (auto &obj : boxarray)
    {
        // INFO("rect: %.2f, %.2f, %.2f, %.2f, confi: %.2f, name: %s", obj.left, obj.top, obj.right, obj.bottom, obj.confidence, cocolabels[obj.class_label]);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), echargercolors[obj.class_label], 3);

        auto name = echargerlabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

        // 可视化结果
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + text_width, obj.top), echargercolors[obj.class_label], -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        if (obj.seg)
        {
            cv::Mat img_clone = image.clone();
            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
            // int count = cv::countNonZero(mask == 255);
            // INFO("count: %d", count);

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

// 生产者线程函数
void SegInference(std::string in_video_url)
{
    cv::Mat image;
    std::string onnx = "../workspace/model/eCharger-v8m.transd.onnx";
    std::string engine = "../workspace/model/eCharger-v8m.transd.engine";
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
    while (cap.read(image))
    {
        try
        {
            // cudaEvent_t start, inf, show;
            // cudaEventCreate(&start);
            // cudaEventCreate(&inf);
            // cudaEventCreate(&show);

            // cudaEventRecord(start);
            seg->inference(image, boxarray);
            // cudaEventRecord(inf);
            show_result(image, boxarray);
            // cudaEventRecord(show);

            // float show_time, inf_time;
            // cudaEventElapsedTime(&inf_time, start, inf);
            // cudaEventElapsedTime(&show_time, inf, show);

            // INFO("inference time: %.3f, show time: %.3f", inf_time, show_time);

            // 加锁队列
            std::unique_lock<std::mutex> lock(mtx);

            // 队列满，等待消费者消费
            cv_producer.wait(lock, []()
                             { bool is_full = img_queue.size() < 30;
            if (!is_full) {
                INFO("Producer is waiting...");
            }
            return is_full; });
            // 图像加入队列
            img_queue.push(image);

            // 通知消费者
            cv_consumer.notify_one();
            INFOD("yolo: the size of image queue : %d ", img_queue.size());
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

    bool end_of_stream = false;
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

            // std::chrono::milliseconds delay(10); // 设置延时时间为30毫秒
            // std::this_thread::sleep_for(delay); // 执行延时
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
    /**
     * YOLOv8m -> 检测时间16ms + 可视化时间 11 ms -> 对应推流 30 fps
     * YOLOv8x -> 检测时间30ms + 可视化时间 11 ms -> 对应推流 20 fps
     */
    // std::string in_url = "rtsp://admin:admin123@192.168.0.213:554/cam/realmonitor?channel=1&subtype=0";
    std::string in_url = "/home/zwy/PyWorkspace/eCharger/TRT_YOLOv8_Server/workspace/images/20230517.mp4";

    int fps = 20, width = 1920, height = 1080, bitrate = 3000000;
    std::string h264profile = "high444";
    std::string out_url = "rtmp://192.168.0.113:1935/myapp/mystream";
    iLogger::set_log_level(iLogger::LogLevel::Info);


    // 创建生产者和消费者线程
    std::thread producer(SegInference, in_url);
    std::thread consumer(video2flv, width, height, fps, bitrate, h264profile, out_url);

    // 等待线程结束
    producer.join();
    consumer.join();
    return 0;
}
