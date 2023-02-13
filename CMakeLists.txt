cmake_minimum_required(VERSION 3.20)
project(openvino_demo)

set(CMAKE_CXX_STANDARD 11)
set(INTEL_OPENVINO_DIR F:/openvino/openvino/build/install/runtime)
set(OpenVINO_DIR ${INTEL_OPENVINO_DIR}/cmake)
set(InferenceEngine_DIR ${INTEL_OPENVINO_DIR}/cmake)
set(ngraph_DIR ${INTEL_OPENVINO_DIR}/cmake)
set(TBB_DIR ${INTEL_OPENVINO_DIR}/3rdparty/tbb/cmake)
set(OpenCV_DIR E:/opencv-4.5.4/opencv/build/x64/vc16/install/x64/vc16/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(OpenCV 4 REQUIRED)

FILE(GLOB SRC_DIR ${PROJECT_SOURCE_DIR}/src/*)

add_executable(openvino_demo asyn_main.cpp ${SRC_DIR})

target_link_libraries(openvino_demo openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})

message(STATUS "opencv_libs:${OpenCV_LIBS}")
message(STATUS "${OpenVINO_FOUND}")
message(STATUS "${OpenVINO_Runtime_FOUND}")
message(STATUS "${OpenVINO_Frontend_ONNX_FOUND}")
message(STATUS "${OpenVINO_VERSION_MAJOR}")
#message(STATUS "${OpenVINO_INCLUDE_PATH}")