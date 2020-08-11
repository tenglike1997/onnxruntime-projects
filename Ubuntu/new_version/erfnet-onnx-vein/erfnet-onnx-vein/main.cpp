#include <QCoreApplication>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>
#include <vector>
#include <stdlib.h>

#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>
//#include <core/providers/tensorrt/tensorrt_provider_factory.h>

using namespace cv;
using namespace std;


//OrtApi *g;
Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };

static constexpr const int width_ = 256;
static constexpr const int height_ = 256;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_*channel> input_image_{};
std::array<float, 1 * 2 * height_ * width_> results_{};
std::array<float, 2> results_extra{};
int result_[height_ * width_]{ 0};

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_{ nullptr };
std::array<int64_t, 4> output_shape_{ 1,2,height_, width_ };


int main()
{
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

    const char* input_names[] = { "input" };
    const char* output_names[] = { "output" };

    Ort::SessionOptions session_option;
    //ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_option, 1));
    //ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
    const ORTCHAR_T* model_path = "/home/tzj/vein-model-pic/ERFNet-vein.onnx";
    Ort::Session session_(env,model_path,session_option);

    const int row = height_;
    const int col = width_;

    Mat img = imread("/home/tzj/vein-model-pic/1.bmp");
    Mat dst(row, col, CV_8UC3);
    Mat dst2;
    resize(img, dst, Size(col, row));
    cvtColor(dst, dst, CV_BGR2RGB);
    //resize(dst2, dst, Size(col, row));

    float* output = input_image_.data();
    fill(input_image_.begin(), input_image_.end(), 0.f);
    Scalar rgb_mean = mean(dst);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                output[c*row*col + i*col + j] = (dst.ptr<uchar>(i)[j*3+c])/255.0 ;
            }
        }
    }

    double timeStart = (double)getTickCount();
    for (int i = 0; i < 1; i++) {
        session_.Run(nullptr, input_names, &input_tensor_, 1, output_names, &output_tensor_,1);
    }
    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "running time ：" << nTime << "sec\n" << endl;


    for (int i = 0; i < height_*width_; i++) {
        results_extra[0] = results_[i];
        results_extra[1] = results_[i + height_ * width_];
        //result_[i] = std::distance(results_extra.begin(), std::max_element(results_extra.begin(), results_extra.end()));
        //result_[i] = results_extra[0] < results_extra[1];
        result_[i] = results_extra[1] > results_extra[0];
    }
    int* result = result_;

    Mat outputimage(height_, width_, CV_8UC1, Scalar(0));
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
            if (result[i * width_ + j] == 0) {
                outputimage.ptr<uchar>(i)[j] = 0;

            }
            if (result[i * width_ + j] == 1) {
                outputimage.ptr<uchar>(i)[j] = 255;
            }
        }
    }

    //Mat outputimage(height_, width_, CV_8UC1, Scalar(0));
    //for (int i = 0; i < height_; i++) {
    //	for (int j = 0; j < width_; j++) {
    //		outputimage.ptr<uchar>(i)[j] = results_[width_*height_+i*width_ + j] * 255.0;
    //	}
    //}

    resize(outputimage, outputimage, Size(row, col));
    imshow("test",outputimage);
    waitKey(0);
    system("pause");
    return 0;
}

