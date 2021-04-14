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


static constexpr const int width_ = 100;
static constexpr const int height_ = 32;
static constexpr const int channel = 1;

std::array<float, 1 * width_ * height_*channel> input_image_{};
std::array<float, 26 * 1 * 37> results_{};
std::array<float, 37> results_extra{};
std::array<float, 26> result_{};

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_{ nullptr };
std::array<int64_t, 3> output_shape_{ 26,1, 37 };

OrtSession* session_ = nullptr;
OrtSessionOptions* session_option;

//Mat Preprocess(const cv::Mat& img)
//{
//	int num_channels_ = 3;
//	Size input_geometry_ = Size(256, 256);
//	//1、通道处理，因为我们如果是Alexnet网络，那么就应该是三通道输入
//	cv::Mat sample;
//	//如果输入图片是一张彩色图片，但是CNN的输入是一张灰度图像，那么我们需要把彩色图片转换成灰度图片
//	if (img.channels() == 3 && num_channels_ == 1)
//		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//	else if (img.channels() == 4 && num_channels_ == 1)
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//	//如果输入图片是灰度图片，或者是4通道图片，而CNN的输入要求是彩色图片，因此我们也需要把它转化成三通道彩色图片
//	else if (img.channels() == 4 && num_channels_ == 3)
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//	else if (img.channels() == 1 && num_channels_ == 3)
//		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//	else
//		sample = img;
//	//缩放处理，因为我们输入的一张图片如果是任意大小的图片，那么我们就应该把它缩放到
//	cv::Mat sample_resized;
//	if (sample.size() != input_geometry_)
//		cv::resize(sample, sample_resized, input_geometry_);
//	else
//		sample_resized = sample;
//	//数据类型处理，因为我们的图片是uchar类型，我们需要把数据转换成float类型
//	cv::Mat sample_float;
//	if (num_channels_ == 3)
//		sample_resized.convertTo(sample_float, CV_32FC3);
//	else
//		sample_resized.convertTo(sample_float, CV_32FC1);
//	//均值归一化
//	cv::Mat sample_normalized;
//	cv::subtract(sample_float, 0, sample_normalized);
//	return sample_normalized;
//}

int main()
{
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

	const char* input_names[] = { "input" };
	const char* output_names[] = { "output" };

	ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_option));
	//ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_option, 1));
	//ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
	//ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
	ORT_THROW_ON_ERROR(OrtCreateSession(env, L"crnn_demo-1008.onnx", session_option, &session_));

	OrtValue *input_tensor_1 = input_tensor_;
	OrtValue *output_tensor_1 = output_tensor_;

	Mat img = imread("C:\\Users\\Administrator.SC-201902250007\\Desktop\\QQ图片20190930164432.png");
	const int row = height_;
	const int col = width_;
	Mat dst(row, col, CV_8UC1);
	//vector<Mat>cutimg;
	//vector<Mat>segcutimg;
	//cutimg = Image_Cut(img);
	//dst = img;
	//Mat dst2;
	//dst2=Preprocess(dst);
	//dst = dst2;
	//dst.convertTo(dst, CV_32FC3);
	//subtract(dst, 0, dst);

	//Mat dst(row, col, CV_8UC3);
	//Mat dst2;
	Mat grayimage;
	cvtColor(img,grayimage, CV_BGR2GRAY);
	resize(grayimage, dst, Size(col, row));

	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	Scalar rgb_mean = mean(dst);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			output[i * col + j] = (dst.ptr<uchar>(i)[j]/255.0-0.5)/0.5;
		}
	}


	double timeStart = (double)getTickCount();
	for (int i = 0; i < 1; i++) {
		OrtRun(session_, nullptr, input_names, &input_tensor_1, 1, output_names, 1, &output_tensor_1);
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time ：" << nTime << "sec\n" << endl;

	//for (int i = 0; i < height_*width_; i++) {
	//	results_extra = results_[i];
	//	//results_extra[1] = results_[i + height_ * width_];
	//	//results_extra[2] = results_[i + height_ * width_ * 2];
	//	//results_extra[3] = results_[i + height_ * width_ * 3];
	//	result_[i] = results_extra*255.0;
	//}
	//int* result = result_;

	//Mat outputimage(height_, width_, CV_8UC1, Scalar(0));
	char alphabet[] = "0123456789abcdefghijklmnopqrstyvwxyz-";
	for (int i = 0; i < 26; i++) {
		for (int j = 0; j < 37; j++) {
			results_extra[j] = results_[i*37+j];
		}
		result_[i] = std::distance(results_extra.begin(), std::max_element(results_extra.begin(), results_extra.end()));
	}
	char text[]="";
	int temp;
	int j = 0;
	for (int i = 0; i < 26; i++) {
		if ((result_[i] != 0) && (!(i > 0 && result_[i - 1] == result_[i]))) {
			temp = result_[i] - 1;
			text[j] = alphabet[temp];
			j++;
		}
	}
	text[j + 1] = '\0';
	cout << text << endl;
	//segcutimg.push_back(outputimage);
	//Mat finalimage(height, width, CV_8UC1, Scalar(0));
	//finalimage = Image_Stitching(segcutimg);
	//imwrite("1.png", outputimage);
	system("pause");
	return 0;
}

