#include <iostream>

#include "cli.hpp"
#include "yolo-nas.hpp"

int main(int argc, char **argv)
{

    Config args = parseCLI(argc, argv);

    cv::Mat img = cv::imread(args.source);

    std::cout << args.modelPath << std::endl;

    YoloNAS model(args.modelPath, args.imgSize, args.gpu, args.scoreTresh, args.iouTresh);


    model.predict(img);


    cv::imshow(args.source, img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}