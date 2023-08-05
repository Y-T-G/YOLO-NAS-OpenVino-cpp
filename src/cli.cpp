/*
Licensed under the MIT License < http://opensource.org/licenses/MIT>.
SPDX - License - Identifier : MIT
Copyright(c) 2023 Mohammed Yasin
Copyright(c) 2023 Wahyu Setianto

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

#include "argparse.hpp"
#include "utils.hpp"
#include "cli.hpp"

Args parseArgs(int argc, char **argv)
{
    argparse::ArgumentParser program("yolo-nas-openvino-cpp");
    program.add_description("YOLO-NAS OpenVINO detection");

    program.add_argument("--model").help("Path to the YOLO-NAS ONNX model.").metavar("MODEL");
    program.add_argument("-i", "--image").help("Path to the image source").metavar("IMAGE");
    program.add_argument("-v", "--video").help("Path to the video source").metavar("VIDEO");

    program.add_argument("--imgsz")
        .help("Model input size")
        .nargs(1)
        .default_value(std::vector<int>{640, 640})
        .scan<'i', int>();
    program.add_argument("--gpu")
        .default_value(false)
        .help("Whether to use GPU");
    program.add_argument("--score-thresh")
        .default_value(0.25f)
        .help("Minimum confidence threshold")
        .scan<'g', float>();
    program.add_argument("--iou-thresh")
        .default_value(0.45f)
        .help("Minimum IoU threshold while applying NMS")
        .scan<'g', float>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << LogError("Parser Error", err.what()) << std::endl;
        std::cerr << program;
        std::abort();
    }

    std::string modelPath = program.get<std::string>("--model");
    bool useGPU = program.get<bool>("--gpu");
    float scoreThresh = program.get<float>("--score-thresh");
    float iouThresh = program.get<float>("--iou-thresh");
    std::vector<int> imgSize = program.get<std::vector<int>>("--imgsz");
    auto imgPath = program.present("-i");
    auto vidPath = program.present("-v");


    exists(modelPath);
    if (imgPath && vidPath)
    {
        std::cerr << LogError("Double Entry", "Please specify either image or video source!") << std::endl;
        std::abort();
    }
    else if (!(imgPath || vidPath))
    {
        std::cerr << LogError("No Entry", "Please input either image or video source!") << std::endl;
        std::abort();
    }

    if (imgSize.size() == 1)
        imgSize.push_back(imgSize[0]);

    Source type;
    std::string source;


    if (imgPath)
    {
        exists(imgPath.value());
        type = IMAGE;
        source = imgPath.value();
    }
    else if (vidPath)
    {
        exists(vidPath.value());
        type = VIDEO;
        source = vidPath.value();
    }

    Args args{modelPath, type, source, imgSize, useGPU, scoreThresh, iouThresh};

    std::string emoji = args.type == IMAGE ? "🖼️" : "📷";
    std::cout << emoji + LogInfo(" Detect", "model=" + args.modelPath);
    std::cout << " source=" + args.source;
    std::cout << " imgsz="
              << "[" << args.imgSize[0] << "," << args.imgSize[1] << "]";
    std::cout << " device=" << (args.gpu ? "true" : "false");
    std::cout << " score-tresh=" << args.scoreThresh;
    std::cout << " iou-thresh=" << args.iouThresh << std::endl;

    return args;
}