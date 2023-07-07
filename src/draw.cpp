#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.hpp"
#include "yolo-nas.hpp"

void drawBoxes(cv::Mat& image, const std::vector<std::vector<Box>>& boxes, float width_ratio, float height_ratio) {
    for (const auto& box_list : boxes) {
        for (const auto& box : box_list) {
            float x1 = box.x1 * width_ratio;
            float y1 = box.y1 * height_ratio;
            float x2 = box.x2 * width_ratio;
            float y2 = box.y2 * height_ratio;

            cv::rectangle(image, cv::Point_<float>(x1, y1), cv::Point_<float>(x2, y2), cv::Scalar(0, 255, 0), 2);

            std::string label = "Class: " + std::to_string(static_cast<int>(box.class_id)) + ", Confidence: " + std::to_string(box.confidence);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(image, cv::Point_<float>(x1, y1 - textSize.height - 5), cv::Point_<float>(x1 + textSize.width, y1), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(image, label, cv::Point_<float>(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
}