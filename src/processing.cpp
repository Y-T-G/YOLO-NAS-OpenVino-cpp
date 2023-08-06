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

PPYoloEPostPredictionCallback::PPYoloEPostPredictionCallback(float score_threshold, float nms_threshold, int nms_top_k, int max_predictions, bool multi_label_per_box)
    : score_threshold(score_threshold), nms_threshold(nms_threshold), nms_top_k(nms_top_k), max_predictions(max_predictions), multi_label_per_box(multi_label_per_box) {}

std::vector<std::vector<Box>> PPYoloEPostPredictionCallback::forward(float* pred_bboxes, float* pred_scores, ov::Shape output_shape_bboxes, ov::Shape output_shape_scores) {
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

std::vector<std::vector<Box>> PPYoloEPostPredictionCallback::_filter_max_predictions(std::vector<std::vector<Box>>& res) const {
    for (auto& im : res) {
        if (im.size() > max_predictions) {
            im.resize(max_predictions);
        }
    }
    return res;
}

std::vector<size_t> PPYoloEPostPredictionCallback::performNMS(const std::vector<Box>& boxes, const std::vector<float>& scores, const std::vector<size_t>& indices, float iou_threshold) const {
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

float PPYoloEPostPredictionCallback::calculateIntersection(const Box& box1, const Box& box2) const {
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);
    return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
}

float PPYoloEPostPredictionCallback::calculateArea(const Box& box) const {
    return (box.x2 - box.x1) * (box.y2 - box.y1);
}

float score_threshold;
float nms_threshold;
int nms_top_k;
int max_predictions;
bool multi_label_per_box;