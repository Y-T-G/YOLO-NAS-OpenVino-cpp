#include <iostream>

#include "cli.hpp"
#include "yolo-nas.hpp"

int main(int argc, char **argv)
{

    Args args = parseArgs(argc, argv);

    cv::Mat img = cv::imread(args.source);

    std::cout << args.modelPath << std::endl;

    YoloNAS model(args.modelPath, args.imgSize, args.gpu, args.scoreThresh, args.iouThresh);


    model.predict(img);


    cv::imshow(args.source, img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}