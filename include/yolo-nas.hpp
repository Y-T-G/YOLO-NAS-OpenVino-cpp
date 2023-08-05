/*
Licensed under the MIT License < http://opensource.org/licenses/MIT>.
SPDX - License - Identifier : MIT
Copyright(c) 2023 Mohammed Yasin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
    void letterbox(cv::Mat &source, cv::Mat &dst, std::vector<float> &ratios);
    void predict(cv::Mat &img);
};