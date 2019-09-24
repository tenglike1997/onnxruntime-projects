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

static constexpr const int width_ = 224;
static constexpr const int height_ = 224;
static constexpr const int channel = 1;

std::array<float, 1 * width_ * height_*channel> input_image_{};
std::array<float, 1 * 1 * 672 * 672> results_{};
std::array<float, 3> results_extra{};
int result_[672 * 672]{ 0 };
Mat outputimageY(672, 672, CV_8UC1, Scalar(0));

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_{ nullptr };
std::array<int64_t, 4> output_shape_{ 1,1,672, 672 };

OrtSession* session_ = nullptr;
OrtSessionOptions* session_option;

int main()
{
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	const char* input_names[] = { "input" };
	const char* output_names[] = { "output" };
	ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_option));
	//ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_option, 1));
	ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
	ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
	ORT_THROW_ON_ERROR(OrtCreateSession(env, L"super_resolution.onnx", session_option, &session_));
	OrtValue *input_tensor_1 = input_tensor_;
	OrtValue *output_tensor_1 = output_tensor_;

	Mat img = imread("..\\..\\test_imgs\\super_resolution\\rawimg.jpg");
	const int row = height_;
	const int col = width_;
	Mat dst(row, col, CV_8UC3);
	Mat dst1;
	
	resize(img, dst, Size(col, row));
	imwrite("..\\..\\test_imgs\\super_resolution\\LowResolution.png", dst);
	
	Mat dstresize;
	resize(dst, dstresize, Size(672, 672));
	imwrite("..\\..\\test_imgs\\super_resolution\\RizeResolution.png", dstresize);

	cvtColor(dst, dst1, COLOR_BGR2YCrCb);
	Mat dstycbcr=dst1;
	vector<Mat> channels1;
	vector<Mat> channels2;
	Mat imageYChannel;
	Mat imageCrChannel;
	Mat imageCbChannel;
	split(dstycbcr, channels1);
	imageYChannel = channels1.at(0);
	imageCrChannel = channels1.at(1);
	imageCbChannel = channels1.at(2);


	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	for (int c = 0;c < 1;c++) {
		for (int i = 0;i < row;i++) {
			for (int j = 0;j < col;j++) {
				output[c*row*col + i*col + j] = (imageYChannel.ptr<uchar>(i)[j]);
			}
		}
	}

	double timeStart = (double)getTickCount();
	for (int i = 0; i < 1000; i++) {
		OrtRun(session_, nullptr, input_names, &input_tensor_1, 1, output_names, 1, &output_tensor_1);
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time ：" << nTime << "sec\n" << endl;

	for (int i = 0; i < 672 * 672; i++) {
		outputimageY.ptr<uchar>(i / 672)[i % 672] = results_[i];
	}

	Mat outputimageCr;
	Mat outputimageCb;
	Mat outputimageY1 = outputimageY;
	resize(imageCrChannel, outputimageCr, Size(672, 672), INTER_CUBIC);
	resize(imageCbChannel, outputimageCb, Size(672, 672), INTER_CUBIC);
	channels2.push_back(outputimageY);
	channels2.push_back(outputimageCr);
	channels2.push_back(outputimageCb);
	Mat outputimage;
	Mat outputimagefinal;
	merge(channels2, outputimage);
	cvtColor(outputimage, outputimagefinal, COLOR_YCrCb2BGR);

	imwrite("..\\..\\test_imgs\\super_resolution\\SuperResolution.png", outputimagefinal);
	system("pause");
	return 0;
}

