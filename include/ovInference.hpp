#include "slog.hpp"
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <chrono>

class ovInference
{
public:
    ovInference() = default;

    bool build();

    bool infer();

private:

};

