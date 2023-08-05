#pragma once

#include <vector>

enum Source
{
    IMAGE,
    VIDEO
};

struct Args
{
    std::string modelPath;
    Source type;
    std::string source;
    std::vector<int> imgSize;
    bool gpu;
    float scoreThresh;
    float iouThresh;
};

Args parseArgs(int argc, char **argv);