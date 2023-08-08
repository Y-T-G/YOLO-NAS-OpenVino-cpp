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

#include "processing.hpp"
#include "yolo-nas.hpp"
#include "utils.hpp"
#include "draw.hpp"


YoloNAS::YoloNAS(std::string modelPath, std::vector<int> imgsz, bool gpu, float score, float iou)
    : postprocessor(score, iou, 1000, 300, false) // define postprocessor
{
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(modelPath);

    core.set_property(ov::cache_dir(".cache"));

    imgSize = imgsz;

    int width = imgSize[0];
    int height = imgSize[1];

    modelInputShape[3] = width;
    modelInputShape[2] = height;

    // preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC");
    ppp.input().preprocess().convert_element_type(ov::element::f32);
    ppp.input().model().set_layout("NCHW");

    // embed above steps in the graph
    model = ppp.build();

    if (gpu)
        try {
        compiled_model = std::make_shared<ov::CompiledModel>(core.compile_model(model, "GPU"));
    }
        catch (const std::runtime_error& err){
            std::cerr << LogWarning("Failed to use GPU. Using CPU instead...", err.what()) << std::endl;
            gpu = false;
        }
    
    if (!gpu) {
        compiled_model = std::make_shared<ov::CompiledModel>(core.compile_model(model, "CPU"));
    }

    infer_request = std::make_shared<ov::InferRequest>(compiled_model -> create_infer_request());

}

void YoloNAS::letterbox(cv::Mat& source, cv::Mat& dst, std::vector<float>& ratios)
{
    // padding image to [n x n] dim
    int maxSize = std::max(source.cols, source.rows);
    int xPad = maxSize - source.cols;
    int yPad = maxSize - source.rows;
    float xRatio = (float)maxSize / (float)modelInputShape[3];
    float yRatio = (float)maxSize / (float)modelInputShape[2];

    cv::copyMakeBorder(source, dst, 0, yPad, 0, xPad, cv::BORDER_CONSTANT); // padding black

    cv::resize(dst, dst, cv::Size(modelInputShape[3], modelInputShape[2]), 0, 0, cv::INTER_NEAREST);

    ratios.push_back(xRatio);
    ratios.push_back(yRatio);
}

void YoloNAS::predict(cv::Mat& img)
{
    cv::Mat imgInput;
    std::vector<float> ratios;
    letterbox(img, imgInput, ratios);

    // Create tensor from image
    float* input_data = (float*)imgInput.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model->input().get_element_type(), compiled_model->input().get_shape(), input_data);

    // Create an infer request for model inference 
    infer_request->set_input_tensor(input_tensor);
    infer_request->infer();

    // Retrieve inference results - bboxes 
    const ov::Tensor& output_tensor_bboxes = infer_request->get_output_tensor(0);
    ov::Shape output_shape_bboxes = output_tensor_bboxes.get_shape();
    float* bboxes = output_tensor_bboxes.data<float>();

    // Retrieve inference results - scores 
    const ov::Tensor& output_tensor_scores = infer_request->get_output_tensor(1);
    ov::Shape output_shape_scores = output_tensor_scores.get_shape();
    float* scores = output_tensor_scores.data<float>();

    // Postprocess predictions
    std::vector<std::vector<Box>> results = postprocessor.forward(bboxes, scores, output_shape_bboxes, output_shape_scores);

    drawBoxes(img, results, ratios[0], ratios[1]);
}
