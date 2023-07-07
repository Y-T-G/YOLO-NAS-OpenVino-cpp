#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct Box {
    float x1, y1, x2, y2, confidence, class_id;
};

#include <draw.hpp>

class YoloNAS
{
private:
    int modelInputShape[4] = { 1, 3, 0, 0 };
    float scoreTresh;
    float iouTresh;
    Colors colors;

public:
    std::shared_ptr<ov::InferRequest> infer_request;
    std::shared_ptr<ov::CompiledModel> compiled_model;
    std::vector<int> imgSize;
    YoloNAS(std::string model_path, std::vector<int> imgsz, bool cuda, float scoreTresh, float iouTresh);
    void preprocess(cv::Mat &source, cv::Mat &dst, std::vector<float> &ratios);
    void predict(cv::Mat &img);
};