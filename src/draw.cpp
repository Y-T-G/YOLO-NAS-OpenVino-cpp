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

#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "draw.hpp"

void drawBoxes(cv::Mat& image, const std::vector<std::vector<Box>>& boxes, float width_ratio, float height_ratio) {
    Colors colorPalette;

    for (const auto& box_list : boxes) {
        for (const auto& box : box_list) {
            float x1 = box.x1 * width_ratio;
            float y1 = box.y1 * height_ratio;
            float x2 = box.x2 * width_ratio;
            float y2 = box.y2 * height_ratio;

            cv::Scalar color = colorPalette.get(static_cast<int>(box.class_id)); // Get the color for the class from the palette

            cv::rectangle(image, cv::Point_<float>(x1, y1), cv::Point_<float>(x2, y2), color, 2);

            std::string label = "Class: " + std::to_string(static_cast<int>(box.class_id)) + ", Confidence: " + std::to_string(box.confidence);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(image, cv::Point_<float>(x1, y1 - textSize.height - 5), cv::Point_<float>(x1 + textSize.width, y1), color, cv::FILLED);
            cv::putText(image, label, cv::Point_<float>(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    } 
}