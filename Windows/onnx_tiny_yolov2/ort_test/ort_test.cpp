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

static constexpr const int width_ = 416;
static constexpr const int height_ = 416;
static constexpr const int channel = 3;

static constexpr const float confidence_threshold = 0.1;
static constexpr const float nms_threshold =0.3f;

std::array<float, 1 * width_ * height_*channel> input_image_{};
std::array<float, 1 * 125 * 13 * 13> results_{};
std::array<float, 20> results_extra{};
vector<Vec4i>True_Point;
vector<int>Type;
vector<int>True_TypeIndex;
float anchors[10] = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };
String classes[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
Scalar colors[20] = { Scalar(255, 0, 0), Scalar(0, 255, 0),Scalar(0,0,255),
Scalar(255,255,0),Scalar(255,0,255), Scalar(0,255,255),
Scalar(255,255,255), Scalar(127,0,0),Scalar(0,127,0),
Scalar(0,0,127),Scalar(127,127,0), Scalar(127,0,127),
Scalar(0,127,127), Scalar(127,127,127),Scalar(127,255,0),
Scalar(127,0,255),Scalar(127,255,255), Scalar(0,127,255),
Scalar(255,127,0), Scalar(0,255,127) };
//int result_[height_ * width_*5]{ 0 };

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_{ nullptr };
std::array<int64_t, 4> output_shape_{ 1,125,13, 13 };

OrtSession* session_ = nullptr;
OrtSessionOptions* session_option;



float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

void nms(
	const std::vector<cv::Rect>& srcRects,
	std::vector<cv::Rect>& resRects,
	float thresh
)
{
	resRects.clear();

	const size_t size = srcRects.size();
	if (!size)
	{
		return;
	}

	// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
	std::multimap<int, size_t> idxs;
	for (size_t i = 0; i < size; ++i)
	{
		idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
	}

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		const cv::Rect& rect1 = srcRects[lastElem->second];
		True_TypeIndex.push_back(lastElem->second);
		resRects.push_back(rect1);

		idxs.erase(lastElem);

		for (auto pos = std::begin(idxs); pos != std::end(idxs); )
		{
			// grab the current rectangle
			const cv::Rect& rect2 = srcRects[pos->second];

			float intArea = (rect1 & rect2).area();
			float unionArea = rect1.area() + rect2.area() - intArea;
			float overlap = intArea / unionArea;

			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > thresh)
			{
				pos = idxs.erase(pos);
			}
			else
			{
				++pos;
			}
		}
	}
}


int main()
{
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	const char* input_names[] = { "image" };
	const char* output_names[] = { "grid" };
	ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_option));
	//ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_option, 1));
	ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
	ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
	ORT_THROW_ON_ERROR(OrtCreateSession(env, L"tiny_yolov2.onnx", session_option, &session_));
	OrtValue *input_tensor_1 = input_tensor_;
	OrtValue *output_tensor_1 = output_tensor_;
	
	Mat img = imread("..\\..\\test_imgs\\detection\\000001.jpg");
	const int row = height_;
	const int col = width_;
	Mat dst(row, col, CV_8UC3);
	Mat dst2;
	resize(img, dst, Size(col, row));
	cvtColor(dst,dst,CV_BGR2RGB);

	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	Scalar rgb_mean = mean(dst);
	for (int c = 0;c < 3;c++) {
		for (int i = 0;i < row;i++) {
			for (int j = 0;j < col;j++) {
				output[c*row*col + i*col + j] = (dst.ptr<uchar>(i)[j * 3 + c]);
			}
		}
	}

	double timeStart = (double)getTickCount();
	for (int i = 0; i < 1000; i++) {
		OrtRun(session_, nullptr, input_names, &input_tensor_1, 1, output_names, 1, &output_tensor_1);
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time ：" << nTime << "sec\n" << endl;

	Vec4i true_point;
	int result_type;
	for (int i = 0; i < 13 * 13; i++) {
		for (int j = 0; j < 5; j++) {
			float sum = 0;
			for (int k = 0; k < 20; k++) {
				results_extra[k] = results_[i + 13 * 13 * (25 * j + k + 5)];
				sum += exp(results_extra[k]);
			}
			result_type = std::distance(results_extra.begin(), std::max_element(results_extra.begin(), results_extra.end()));
			float probability = exp(results_extra[result_type]) / sum;
			if (sigmoid(results_[i + 13 * 13 * (25 * j + 4)])*probability >= confidence_threshold) {
				true_point[0] = (sigmoid(results_[i + 13 * 13 * (25 * j)]) + i % 13)*32.0;
				true_point[1] = (sigmoid(results_[i + 13 * 13 * (25 * j + 1)]) + i / 13)*32.0;
				true_point[2] = exp(results_[i + 13 * 13 * (25 * j + 2)])*anchors[2 * j] * 32.0;
				true_point[3] = exp(results_[i + 13 * 13 * (25 * j + 3)])*anchors[2 * j + 1] * 32.0;
				True_Point.push_back(true_point);
				Type.push_back(result_type);
			}
		}
	}
	vector<Rect>srcRects;
	vector<Rect>resRects;
	Rect rect;
	for (int i = 0;i < True_Point.size();i++) {
		rect = Rect(True_Point[i][0] - True_Point[i][2] / 2.0, True_Point[i][1] - True_Point[i][3] / 2.0, True_Point[i][2], True_Point[i][3]);
		srcRects.push_back(rect);
	}
	nms(srcRects, resRects, nms_threshold);
	for (int i = 0;i < resRects.size();i++) {
		rectangle(dst, resRects[i], Scalar(colors[Type[True_TypeIndex[i]]]), 3, 1, 0);
		cout << Type[True_TypeIndex[i]] << endl;
		putText(dst,classes[Type[True_TypeIndex[i]]],Point(resRects[i].x+5,resRects[i].y+13), FONT_HERSHEY_COMPLEX,0.5,Scalar(colors[Type[True_TypeIndex[i]]]),1,8);
	}
	imwrite("..\\..\\test_imgs\\detection\\test.jpg", dst);
	system("pause");
    return 0;
}

