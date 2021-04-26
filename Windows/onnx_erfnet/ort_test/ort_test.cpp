#include "stdafx.h"
#include <windows.h>
#include <windowsx.h>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>
#include <tensorrt_provider_factory.h>
#include <mkldnn_provider_factory.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <vector>
#include <stdlib.h> 
#include <iostream> 

using namespace cv;
using namespace std;


#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "onnxruntime.lib")

Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };

static constexpr const int width_ = 640;
static constexpr const int height_ = 480;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_*channel> input_image_{};
std::array<float, 1 * 4 * height_ * width_> results_{};
std::array<float, 4> results_extra{};
int result_[4*height_ * width_]{ 0};

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_{ nullptr };
std::array<int64_t, 4> output_shape_{ 1,4,height_, width_ };

OrtSession* session_ = nullptr;
OrtSessionOptions* session_option;

int main()
{
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

	const char* input_names[] = { "actual_input_1" };
	const char* output_names[] = { "output1" };
	
	ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_option));
	//ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_option, 1));
	ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
	ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
	ORT_THROW_ON_ERROR(OrtCreateSession(env, L"erfnet.onnx", session_option, &session_));

	OrtValue *input_tensor_1 = input_tensor_;
	OrtValue *output_tensor_1 = output_tensor_;

	Mat img = imread("..\\..\\test_imgs\\segmentation\\00004.png");
	const int row = height_;
	const int col = width_;
	Mat dst(row, col, CV_8UC3);
	Mat dst2;
	resize(img, dst, Size(col, row));
	cvtColor(dst, dst, CV_BGR2RGB);

	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	Scalar rgb_mean = mean(dst);
	for (int c = 0;c < 3;c++) {
		for (int i = 0;i < row;i++) {
			for (int j = 0;j < col;j++) {

				output[c*row*col + i*col + j] = (dst.ptr<uchar>(i)[j * 3 + c])/255.0;
			}
		}
	}

	double timeStart = (double)getTickCount();
	for (int i = 0; i < 1000; i++) {
		OrtRun(session_, nullptr, input_names, &input_tensor_1, 1, output_names, 1, &output_tensor_1);
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time ：" << nTime << "sec\n" << endl;

	for (int i = 0; i < height_*width_; i++) {
		results_extra[0] = results_[i];
		results_extra[1] = results_[i + height_ * width_];
		results_extra[2] = results_[i + height_ * width_ * 2];
		results_extra[3] = results_[i + height_ * width_ * 3];
		result_[i] = std::distance(results_extra.begin(), std::max_element(results_extra.begin(), results_extra.end()));
	}
	int* result = result_;

	Mat outputimage(height_, width_, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0;i < height_;i++) {
		for (int j = 0;j < width_;j++) {
			if (result[i * width_ + j] == 0) {
				outputimage.ptr<uchar>(i)[j * 3] = 255;
				outputimage.ptr<uchar>(i)[j * 3+1] = 0;
				outputimage.ptr<uchar>(i)[j * 3+2] = 0;
			}
			if (result[i * width_ + j] == 1) {
				outputimage.ptr<uchar>(i)[j * 3] = 0;
				outputimage.ptr<uchar>(i)[j * 3 + 1] = 255;
				outputimage.ptr<uchar>(i)[j * 3 + 2] = 0;
			}
			if (result[i * width_ + j] == 2) {
				outputimage.ptr<uchar>(i)[j * 3] = 0;
				outputimage.ptr<uchar>(i)[j * 3 + 1] = 0;
				outputimage.ptr<uchar>(i)[j * 3 + 2] = 255;
			}
			if (result[i * width_ + j] == 3) {
				outputimage.ptr<uchar>(i)[j * 3] = 255;
				outputimage.ptr<uchar>(i)[j * 3 + 1] = 255;
				outputimage.ptr<uchar>(i)[j * 3 + 2] = 0;
			}
		}
	}
	imwrite("4.png", outputimage);
	system("pause");
    return 0;
}

