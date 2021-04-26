#include "stdafx.h"
#include <windows.h>
#include <windowsx.h>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>
//#include <tensorrt_provider_factory.h>
//#include <mkldnn_provider_factory.h>

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

using namespace cv;
using namespace std;


//Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };

static constexpr const int width_ = 1200;
static constexpr const int height_ = 1200;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_*channel> input_image_{};
//std::array<float, 1 * 2 * height_ * width_> results_{};
float *results_;
float *results_0;
int *results_1;
float *results_2;
std::array<float, 2> results_extra{};
//int result_[height_ * width_]{ 0 };

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_0{ nullptr };
Ort::Value output_tensor_1{ nullptr };
Ort::Value output_tensor_2{ nullptr };
vector<Ort::Value>output_tensor_;
//vector<OrtValue*> output_tensor_(3);
//output_tensor_[0] = output_tensor_0;
//output_tensor_[1] = output_tensor_1;
//output_tensor_[2] = output_tensor_2;
std::array<int64_t, 4> output_shape_{ 1,2,height_, width_ };

String classes[80] = {
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
};  //类别
Scalar colors[20] = { Scalar(255, 0, 0), Scalar(0, 255, 0),Scalar(0,0,255),
Scalar(255,255,0),Scalar(255,0,255), Scalar(0,255,255),
Scalar(255,255,255), Scalar(127,0,0),Scalar(0,127,0),
Scalar(0,0,127),Scalar(127,127,0), Scalar(127,0,127),
Scalar(0,127,127), Scalar(127,127,127),Scalar(127,255,0),
Scalar(127,0,255),Scalar(127,255,255), Scalar(0,127,255),
Scalar(255,127,0), Scalar(0,255,127) };  //颜色

static constexpr const float confidence_threshold = 0.3;  //置信度阈值
static constexpr const float nms_threshold = 0.5f; //nms置信度

void nms(
	const std::vector<cv::Rect>& srcRects,
	std::vector<cv::Rect>& resRects,
	std::vector<int>& resIndexs,
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
		resIndexs.push_back(lastElem->second);
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
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	//output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	//g_ort->CreateTensorWithDataAsOrtValue(allocator_info, results_.data(), results_.size()*sizeof(float), output_shape_.data(), output_shape_.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&output_tensor_);

	const char* input_names[] = { "image" };
	vector<const char*>output_names(3);
	output_names[0] = { "bboxes" };
	output_names[1] = { "labels" };
	output_names[2] = { "scores" };

	Ort::SessionOptions session_option;
	//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_option, 1));
	//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
	//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
	session_option.SetIntraOpNumThreads(1);
	session_option.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	Ort::Session session_(env, L"C:\\Users\\admin\\Desktop\\onnx_ssd\\ort_test\\ssd.onnx", session_option);

	const int row = height_;
	const int col = width_;

	Mat img = imread("C:\\Users\\admin\\Desktop\\onnx测试图片\\VOC\\000011.jpg");
	Mat dst(row, col, CV_8UC3);
	resize(img, dst, Size(col, row));
	cvtColor(dst, dst, CV_BGR2RGB);

	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	Scalar rgb_mean = mean(dst);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (c == 0) {
					output[c*row*col + i*col + j] = ((dst.ptr<uchar>(i)[j * 3 + c]) / 255.0 - 0.406) / 0.225;
				}
				if (c == 1) {
					output[c*row*col + i*col + j] = ((dst.ptr<uchar>(i)[j * 3 + c]) / 255.0 - 0.456) / 0.224;
				}
				if (c == 2) {
					output[c*row*col + i*col + j] = ((dst.ptr<uchar>(i)[j * 3 + c]) / 255.0 - 0.485) / 0.229;
				}

			}
		}
	}


	double timeStart = (double)getTickCount();
	for (int i = 0; i < 1; i++) {
		output_tensor_ = session_.Run(nullptr, input_names, &input_tensor_, 1, output_names.data(), 3);
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time ：" << nTime << "sec\n" << endl;

	results_0 = output_tensor_[0].GetTensorMutableData<float>();
	Ort::TensorTypeAndShapeInfo info_0 = output_tensor_[0].GetTensorTypeAndShapeInfo();
	size_t outlength_0;
	outlength_0 = info_0.GetElementCount();
	size_t out_dimensions_0;
	out_dimensions_0 = info_0.GetDimensionsCount();

	results_1 = output_tensor_[1].GetTensorMutableData<int>();
	Ort::TensorTypeAndShapeInfo info_1 = output_tensor_[1].GetTensorTypeAndShapeInfo();
	size_t outlength_1;
	outlength_1 = info_1.GetElementCount();
	size_t out_dimensions_1;
	out_dimensions_1 = info_1.GetDimensionsCount();

	results_2 = output_tensor_[2].GetTensorMutableData<float>();
	Ort::TensorTypeAndShapeInfo info_2 = output_tensor_[2].GetTensorTypeAndShapeInfo();
	size_t outlength_2;
	outlength_2 = info_2.GetElementCount();
	size_t out_dimensions_2;
	out_dimensions_2 = info_2.GetDimensionsCount();

	//float* location = new float[outlength_0];
	vector<Vec4f>location(outlength_1);
	for (int i = 0; i < outlength_1; i++) {
		location[i][0] = results_0[i * 4] * 1200;
		location[i][1] = results_0[i * 4 + 1] * 1200;
		location[i][2] = results_0[i * 4 + 2] * 1200;
		location[i][3] = results_0[i * 4 + 3] * 1200;
	}

	//int* label = new int[outlength_1];
	vector<int>label(outlength_1);
	for (int i = 0; i < outlength_1; i++) {
		label[i] = results_1[i * 2];
	}

	//float* confidence = new float[outlength_2];
	vector<float>confidence(outlength_2);
	for (int i = 0; i < outlength_2; i++) {
		confidence[i] = results_2[i];
	}


	vector<Rect>srcRects;
	vector<Rect>resRects;
	vector<int>srcLabels;
	vector<int>resLabels;
	vector<int>resIndexs;
	Rect rect;
	for (int i = 0; i < outlength_1; i++) {
		rect = Rect(location[i][0], location[i][1], location[i][2] - location[i][0], location[i][3] - location[i][1]);
		if (confidence[i] > confidence_threshold) {
			srcRects.push_back(rect);
			srcLabels.push_back(label[i]);
		}
	}
	nms(srcRects, resRects, resIndexs, nms_threshold);
	for (int i = 0; i < resIndexs.size(); i++) {
		resLabels.push_back(srcLabels[resIndexs[i]]);
	}
	for (int i = 0; i < resRects.size(); i++) {
		rectangle(dst, resRects[i], Scalar(colors[resLabels[i] % 4]), 3, 1, 0);
		//cout << Type[True_TypeIndex[i]] << endl;
		putText(dst, classes[resLabels[i] - 1], Point(resRects[i].x + 5, resRects[i].y + 13), FONT_HERSHEY_COMPLEX, 0.5, Scalar(colors[resLabels[i] % 4]), 1, 8);
	} //图片上画框标类别

	imwrite("C:\\Users\\admin\\Desktop\\onnx测试图片\\VOC\\test.jpg",dst);
	system("pause");
	return 0;
}
