#include <iostream>

#include "cli.hpp"
#include "yolo-nas.hpp"

#include <chrono>

int predictImage(YoloNAS model, Args args) {

	cv::Mat img = cv::imread(args.source);

	model.predict(img);

	cv::imshow(args.source, img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

int predictVideo(YoloNAS model, Args args) {

	cv::VideoCapture cap = cv::VideoCapture(args.source);
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
	float latency;

	while (cap.isOpened()) {
		cv::Mat frame;

		cap >> frame;

		if (!frame.empty()) {
			begin = std::chrono::steady_clock::now();
			model.predict(frame);
			end = std::chrono::steady_clock::now();
			cv::imshow(args.source, frame);

			latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			std::cout << "Latency = " << latency << "ms\t";
			std::cout << "FPS = " << 1000.0 / latency << std::endl;

		}

		if (cv::waitKey(30) == 27)
		{
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}

int main(int argc, char** argv)
{

	Args args = parseArgs(argc, argv);

	YoloNAS model(args.modelPath, args.imgSize, args.gpu, args.scoreThresh, args.iouThresh);

	if (args.type == IMAGE) {
		predictImage(model, args);
	}

	else  if (args.type == VIDEO) {
		predictVideo(model, args);
	}

	return 0;
}