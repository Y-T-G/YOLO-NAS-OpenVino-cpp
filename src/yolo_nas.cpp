#include "yolo-nas.hpp"
#include "utils.hpp"

#include <openvino/openvino.hpp>
#include <vector>
#include <algorithm>

class PPYoloEPostPredictionCallback {
public:
    PPYoloEPostPredictionCallback(float score_threshold, float nms_threshold, int nms_top_k, int max_predictions, bool multi_label_per_box = true)
        : score_threshold(score_threshold), nms_threshold(nms_threshold), nms_top_k(nms_top_k), max_predictions(max_predictions), multi_label_per_box(multi_label_per_box) {}

    std::vector<std::vector<Box>> forward(float* pred_bboxes, float* pred_scores, ov::Shape output_shape_bboxes, ov::Shape output_shape_scores) {
        std::vector<std::vector<Box>> nms_result;

            std::vector<Box> filtered_boxes;

            // Filter all predictions by self.score_threshold
            // TODO: Add multi_label support
            if (multi_label_per_box) {
                for (size_t i = 0; i < output_shape_scores.at(1); i++) {
                    for (size_t j = 0; j < output_shape_scores.at(2); j++) {
                        if (*(pred_scores + (i * output_shape_scores.at(1)) + j) > score_threshold) {
                            Box box;
                            auto bbox_begin = pred_bboxes + (i * output_shape_bboxes.at(2));
                            box.x1 = *(bbox_begin);
                            box.y1 = *(bbox_begin + 1);
                            box.x2 = *(bbox_begin + 2);
                            box.y2 = *(bbox_begin + 3);
                            box.confidence = *(pred_scores + (i * output_shape_scores.at(1)) + j);
                            box.class_id = static_cast<float>(j);
                            filtered_boxes.push_back(box);
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < output_shape_scores.at(1); i++) {
                    auto score_begin = pred_scores + (i * output_shape_scores.at(2));
                    auto score_end = score_begin + output_shape_scores.at(2);
                    auto max_el = std::max_element(score_begin, score_end);
                    float max_score = *max_el;
                    auto bbox_begin = pred_bboxes + (i * output_shape_bboxes.at(2));
                    size_t max_index = std::distance(score_begin, max_el);
                    if (max_score >= score_threshold) {
                        Box box;
                        box.x1 = *(bbox_begin);
                        box.y1 = *(bbox_begin + 1);
                        box.x2 = *(bbox_begin + 2);
                        box.y2 = *(bbox_begin + 3);
                        box.confidence = max_score;
                        box.class_id = static_cast<float>(max_index);
                        filtered_boxes.push_back(box);
                    }
                }
            }

            // Sort predictions by confidence score
            std::sort(filtered_boxes.begin(), filtered_boxes.end(), [](const Box& a, const Box& b) {
                return a.confidence > b.confidence;
                });

            // Filter all predictions by self.nms_top_k
            if (filtered_boxes.size() > nms_top_k) {
                filtered_boxes.resize(nms_top_k);
            }

            // NMS
            std::vector<float> scores;
            std::vector<size_t> indices;
            for (const auto& box : filtered_boxes) {
                scores.push_back(box.confidence);
                indices.push_back(static_cast<size_t>(box.class_id));
            }
            std::vector<size_t> idx_to_keep = performNMS(filtered_boxes, scores, indices, nms_threshold);

            std::vector<Box> final_boxes;
            for (const auto& idx : idx_to_keep) {
                final_boxes.push_back(filtered_boxes[idx]);
            }
            nms_result.push_back(final_boxes);

        return _filter_max_predictions(nms_result);
    }

private:
    std::vector<std::vector<Box>> _filter_max_predictions(std::vector<std::vector<Box>>& res) const {
        for (auto& im : res) {
            if (im.size() > max_predictions) {
                im.resize(max_predictions);
            }
        }
        return res;
    }

    std::vector<size_t> performNMS(const std::vector<Box>& boxes, const std::vector<float>& scores, const std::vector<size_t>& indices, float iou_threshold) const {
        std::vector<size_t> keep;
        std::vector<bool> suppressed(boxes.size(), false);

        for (size_t i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) {
                continue;
            }

            keep.push_back(i);

            for (size_t j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) {
                    continue;
                }

                float intersection = calculateIntersection(boxes[i], boxes[j]);
                float iou = intersection / (calculateArea(boxes[i]) + calculateArea(boxes[j]) - intersection);

                if (iou > iou_threshold && indices[i] == indices[j]) {
                    suppressed[j] = true;
                }
            }
        }

        return keep;
    }

    float calculateIntersection(const Box& box1, const Box& box2) const {
        float x1 = std::max(box1.x1, box2.x1);
        float y1 = std::max(box1.y1, box2.y1);
        float x2 = std::min(box1.x2, box2.x2);
        float y2 = std::min(box1.y2, box2.y2);
        return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    }

    float calculateArea(const Box& box) const {
        return (box.x2 - box.x1) * (box.y2 - box.y1);
    }

    float score_threshold;
    float nms_threshold;
    int nms_top_k;
    int max_predictions;
    bool multi_label_per_box;
};

YoloNAS::YoloNAS(std::string modelPath, std::vector<int> imgsz, bool gpu, float score, float iou)
{
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(modelPath);

    // set model cache directory
    core.set_property(ov::cache_dir(".cache"));

    imgSize = imgsz;

    int width = imgSize[0];
    int height = imgSize[1];

    modelInputShape[3] = width;
    modelInputShape[2] = height;

    scoreTresh = score;
    iouTresh = iou;

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

    cv::resize(dst, dst, cv::Size(modelInputShape[3], modelInputShape[2]), 0, 0, cv::INTER_AREA);

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

    PPYoloEPostPredictionCallback postprocessor(scoreTresh, iouTresh, 1000, 300, false);

    std::vector<std::vector<Box>> results = postprocessor.forward(bboxes, scores, output_shape_bboxes, output_shape_scores);


    drawBoxes(img, results, ratios[0], ratios[1]);
}
