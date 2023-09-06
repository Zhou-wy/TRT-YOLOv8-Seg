#include "yolov8_seg.hpp"
#include "decoder_kernel.cuh"
#include "segment.hpp"

#include "../src/TrtLib/common/infer_controller.hpp"
#include "../src/TrtLib/common/ilogger.hpp"
#include "../src/TrtLib/infer/trt_infer.hpp"
#include "../src/TrtLib/common/cuda_tools.cuh"
#include "../src/TrtLib/common/preprocess_kernel.cuh"

#include <opencv2/opencv.hpp>
#include <future>

InstanceSegmentMap::InstanceSegmentMap(int _width, int _height)
{
    this->width = _width;
    this->height = _height;
    checkCudaRuntime(cudaMallocHost(&this->data, _width * _height));
}

InstanceSegmentMap::~InstanceSegmentMap()
{
    if (this->data)
    {
        checkCudaRuntime(cudaFreeHost(this->data));
        this->data = nullptr;
    }
    this->width = 0;
    this->height = 0;
}

namespace YOLOv8Seg
{

    static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
    {
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    struct AffineMatrix
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const cv::Size &from, const cv::Size &to)
        {
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat()
        {
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    using ControllerImpl = InferController<
        cv::Mat,                      // input
        BoxSeg,                       // output
        std::tuple<std::string, int>, // start param
        AffineMatrix                  // additional
        >;

    class InferImpl : public SegInfer, public ControllerImpl
    {
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl()
        {
            stop();
        }

        virtual bool startup(
            const std::string &file, Task task, int gpuid,
            float confidence_threshold, float nms_threshold,
            NMSMethod nms_method, int max_objects,
            bool use_multi_preprocess_stream)
        {
            if (task == Task::det)
            {
                normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            }
            else if (task == Task::seg)
            {
                normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
                // num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
            }
            else if (task == Task::clf)
            {
            }
            else if (task == Task::pos)
            {
            }
            else
            {
                INFOE("Unsupport type %d", task);
            }

            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_ = nms_threshold;
            nms_method_ = nms_method;
            max_objects_ = max_objects;
            task_ = task;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        void worker(std::promise<bool> &result) override
        {
            std::string file = std::get<0>(start_param_);
            int gpuid = std::get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if (engine == nullptr)
            {
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();
            const int MAX_IMAGE_BBOX = max_objects_; // 1024
            const int NUM_BOX_ELEMENT = 8;           // left, top, right, bottom, confidence, class, keepflag, row_index(output)

            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_boxarray_device(TRT::DataType::Float);
            TRT::Tensor output_segment_device(TRT::DataType::Float);

            int max_batch_size = engine->get_max_batch_size(); // max_batch_size = 10

            auto input = engine->tensor("images"); //  {10 x 3 x 640 x 640}

            /**
             * -1 x 116 x 8400
             * 116 -> cx, cy, w, h, 80 class confidence, 32 mask weight
             * 8400 -> 3 anchar
             */
           

            /**
             * -1 x 32 x 160 x 160
             * 32 个 160 * 160 prob map
             * mask = crop([cx, cy, w, h], sigmod(sum(32 weight -> 32 * 160 * 160, dim = 0)))
             */
            int num_classes = 0;
            if (this->task_ == Task::seg)
            {
                bbox_head_dims_ = engine->tensor("output0");
                segment_head_dims_ = engine->tensor("output1"); // -1 x 32 x 160 x 160
                num_classes = bbox_head_dims_->size(2) - 4 - segment_head_dims_->size(1);
            }
            else if (this->task_ == Task::det)
            {
                bbox_head_dims_ = engine->tensor("output0");
                num_classes = bbox_head_dims_->size(2) - 4;
            }

            INFO("num classes : %d", num_classes);
            input_width_ = input->size(3);
            input_height_ = input->size(2);

            tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = engine->get_stream();
            gpu_ = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            affin_matrix_device.resize(max_batch_size, 8).to_gpu();
            output_boxarray_device.resize(max_batch_size, 32 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

            std::vector<Job> fetch_jobs;
            while (get_jobs_and_wait(fetch_jobs, max_batch_size))
            {

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    auto &job = fetch_jobs[ibatch];
                    auto &mono = job.mono_tensor->data();

                    if (mono->get_stream() != stream_)
                    {
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }

                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(),
                                                      6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                output_boxarray_device.to_gpu(false);
                // std::vector<AffineMatrix> affine_matrixs(infer_batch_size);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {

                    auto &job = fetch_jobs[ibatch];
                    float *image_based_output = bbox_head_dims_->gpu<float>(ibatch);
                    float *output_array_ptr = output_boxarray_device.gpu<float>(ibatch);
                    auto affine_matrix = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(image_based_output, bbox_head_dims_->size(1), num_classes,
                                          bbox_head_dims_->size(2), confidence_threshold_,
                                          nms_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX,
                                          NUM_BOX_ELEMENT, Task::seg, stream_);
                }

                checkCudaRuntime(cudaStreamSynchronize(stream_));

                output_boxarray_device.to_cpu();
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    float *parray = output_boxarray_device.cpu<float>(ibatch);
                    int count = std::min(MAX_IMAGE_BBOX, (int)*parray);
                    auto &job = fetch_jobs[ibatch];
                    auto &image_based_boxes = job.output;

                    for (int i = 0; i < count; ++i)
                    {
                        float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                        int label = pbox[5];
                        int keepflag = pbox[6];
                        if (keepflag == 1)
                        {
                            Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);

                            if (task_ == Task::seg)
                            {
                                int row_index = pbox[7];
                                auto segment_head_dims_ = engine->tensor("output1"); // -1 x 32 x 160 x 160
                                int mask_dim = segment_head_dims_->size(1);

                                float left, top, right, bottom;
                                float *i2d = job.additional.i2d;
                                affine_project(i2d, pbox[0], pbox[1], &left, &top);
                                affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

                                float box_width = right - left;
                                float box_height = bottom - top;

                                float scale_to_predict_x = segment_head_dims_->size(3) / (float)input_width_;
                                float scale_to_predict_y = segment_head_dims_->size(2) / (float)input_height_;

                                int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                                int mask_out_height = box_height * scale_to_predict_y + 0.5f;

                                if (mask_out_width > 0 && mask_out_height > 0)
                                {

                                    int bytes_of_mask_out = mask_out_width * mask_out_height;

                                    std::shared_ptr<InstanceSegmentMap> mask_out_host = std::make_shared<InstanceSegmentMap>(
                                        mask_out_width, mask_out_height);
                                    result_object_box.seg = std::make_shared<InstanceSegmentMap>(pbox[2] - pbox[0],
                                                                                                 pbox[3] - pbox[1]);

                                    output_segment_device.resize(max_batch_size, bytes_of_mask_out).to_gpu();

                                    float *mask_head_predict = segment_head_dims_->gpu<float>(ibatch);
                                    float *mask_weights = bbox_head_dims_->gpu<float>(ibatch) +
                                                          (ibatch * bbox_head_dims_->size(1) + row_index) *
                                                              bbox_head_dims_->size(2) +
                                                          num_classes + 4;

                                    unsigned char *mask_out_device = output_segment_device.gpu<unsigned char>(ibatch);

                                    decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y,
                                                       mask_weights, mask_head_predict,
                                                       segment_head_dims_->size(3), segment_head_dims_->size(2),
                                                       mask_out_device,
                                                       mask_dim, mask_out_width, mask_out_height, stream_);

                                    checkCudaRuntime(
                                        cudaMemcpyAsync(mask_out_host->data, mask_out_device, bytes_of_mask_out,
                                                        cudaMemcpyDeviceToHost, stream_));

                                    // resize to source image size

                                    CUDAKernel::resize_bilinear(mask_out_host->data, sizeof(uint8_t) * mask_out_width,
                                                                mask_out_width, mask_out_height,
                                                                result_object_box.seg->data, int(pbox[2] - pbox[0]),
                                                                int(pbox[3] - pbox[1]), stream_);
                                    checkCudaRuntime(cudaStreamSynchronize(stream_));
                                }
                            }
                            image_based_boxes.emplace_back(result_object_box);
                        }
                    }
                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        bool preprocess(Job &job, const cv::Mat &image) override
        {
            if (tensor_allocator_ == nullptr)
            {
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            if (image.empty())
            {
                INFOE("Image is empty");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if (job.mono_tensor == nullptr)
            {
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto &tensor = job.mono_tensor->data();
            TRT::CUStream preprocess_stream = nullptr;

            if (tensor == nullptr)
            {
                // not init
                tensor = std::make_shared<TRT::Tensor>();
                tensor->set_workspace(std::make_shared<TRT::MixMemory>());

                if (use_multi_preprocess_stream_)
                {
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));

                    // owner = true, stream needs to be free during deconstruction
                    tensor->set_stream(preprocess_stream, true);
                }
                else
                {
                    preprocess_stream = stream_;

                    // owner = false, tensor ignored the stream
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            cv::Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);

            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            size_t size_matrix = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace = tensor->get_workspace();
            uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_matrix + size_image);
            float *affine_matrix_device = (float *)gpu_workspace;
            uint8_t *image_device = size_matrix + gpu_workspace;

            uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_matrix + size_image);
            float *affine_matrix_host = (float *)cpu_workspace;
            uint8_t *image_host = size_matrix + cpu_workspace;

            // checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            //  speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(
                cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i),
                                             cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device, image.cols * 3, image.cols, image.rows,
                tensor->gpu<float>(), input_width_, input_height_,
                affine_matrix_device, 114,
                normalize_, preprocess_stream);
            return true;
        }

        std::vector<std::shared_future<BoxSeg>> commits(const std::vector<cv::Mat> &images) override
        {
            return ControllerImpl::commits(images);
        }

        std::shared_future<BoxSeg> commit(const cv::Mat &image) override
        {
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_ = 0;
        int max_objects_ = 1024;
        NMSMethod nms_method_ = NMSMethod::FastGPU;
        TRT::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
        Task task_ = Task::seg;
        int num_classes_;
        std::shared_ptr<TRT::Tensor> segment_head_dims_;
        std::shared_ptr<TRT::Tensor> bbox_head_dims_;
    };

    std::shared_ptr<SegInfer>
    create_seg_infer(const std::string &engine_file, Task task, int gpuid, float confidence_threshold,
                     float nms_threshold, NMSMethod nms_method, int max_objects, bool use_multi_preprocess_stream)
    {
        std::shared_ptr<InferImpl> instance(new InferImpl());
        if (!instance->startup(
                engine_file, task, gpuid, confidence_threshold,
                nms_threshold, nms_method, max_objects, use_multi_preprocess_stream))
        {
            instance.reset();
        }
        return instance;
    }

};