#include "slog.hpp"
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <chrono>

void printInputAndOutputsInfo(const ov::Model &network) {
    slog::info << "model name: " << network.get_friendly_name() << slog::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> input : inputs) {
        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        const ov::element::Type type = input.get_element_type();
        const ov::Shape shape = input.get_shape();
        slog::info << "        input name: " << name << "\t"\
 << "        input type: " << type << "\t"\
 << "        input shape: " << shape << slog::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> output : outputs) {
        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        const ov::element::Type type = output.get_element_type();
        const ov::Shape shape = output.get_shape();
        slog::info << "        output name: " << name << "\t"\
 << "        output type: " << type << "\t"\
 << "        output shape: " << shape << slog::endl;
    }
}

int main() {
    try {
        //!< set parameters
        std::vector<std::string> image_paths = {"../samples/3.png",
                                                "../samples/2.png",
                                                "../samples/6_7.png",
                                                "../samples/4.png"};
        std::string model_path = "../ov_models/2023_2_1_hj_num_2.xml";
        std::string device_name = "GPU";
        std::string labeName[]{"0", "1", "2", "3", "4", "6"};

        //!< preparation
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        slog::info << "Loading model files:" << model_path << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        printInputAndOutputsInfo(*model);
        OPENVINO_ASSERT(model->inputs().size() == 1, "Model with 1 input");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Model with 1 output");

        //!< build model
        const ov::Layout tensor_layout{"NHWC"};
        ov::element::Type input_type = ov::element::f32;

        ov::preprocess::PrePostProcessor ppp(model);

        ppp.input().tensor().set_element_type(input_type).\
        set_layout(tensor_layout);

        ppp.input().model().set_layout("NCHW");

        ppp.output().tensor().set_element_type(ov::element::f32);

        model = ppp.build();

        //!< obtain the valid data
        ov::Shape input_shape = model->input().get_shape();
        const size_t input_width = input_shape[ov::layout::width_idx(tensor_layout)];
        const size_t input_height = input_shape[ov::layout::height_idx(tensor_layout)];
        const size_t input_channels = input_shape[ov::layout::channels_idx(tensor_layout)];
//        slog::info << input_width << "  " << input_height << "  " << input_channels << slog::endl;
        std::vector<std::shared_ptr<float>> input_datas;
        for (const std::string &image_path : image_paths) {
            //!< load image by opencv
            cv::Mat image = cv::imread(image_path);
            cv::resize(image, image, cv::Size(input_width, input_height));
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//            cv::imshow("image",image);
            image.convertTo(image, CV_32FC1);
            image = image / 255.0f;
//            cv::waitKey(0);

            //!< process input
            std::shared_ptr<float> input_data;
            input_data.reset(new float[input_width * input_height * input_channels],
                             std::default_delete<float[]>());
            cv::Mat resized(cv::Size(input_width, input_height), image.type(), input_data.get());
//            slog::info<<image.type()<<slog::endl;
//            cv::resize(image,resized,cv::Size(input_width,input_height));
            image.copyTo(resized);
            if (input_data != nullptr)
                input_datas.push_back(input_data);
        }
        const size_t batchSize = input_datas.size();
//        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

//        slog::info << input_datas.size() << slog::endl;
        ov::set_batch(model, batchSize);
        printInputAndOutputsInfo(*model);
        //        using tensor_itype = ov::fundamental_type_for<ov::element::Type_t::f32>;//float here

        //!< Combine data to Tensor, compile model
        auto start = std::chrono::high_resolution_clock::now();
//        double t = (double)cv::getTickCount();

        ov::CompiledModel compiledModel = core.compile_model(model, device_name);

        ov::InferRequest inferRequest = compiledModel.create_infer_request();

        ov::Tensor input_tensor = inferRequest.get_input_tensor();
        slog::info<<input_tensor.get_shape()<<slog::endl;

        for (size_t i = 0; i < batchSize; i++) {
            const size_t data_size = ov::shape_size(model->input().get_shape()) / batchSize;
//            slog::info << data_size << slog::endl;
            std::memcpy(input_tensor.data<float>() + i * data_size, input_datas[i].get(),
                        data_size*4);
        }
        slog::info<<"Start Infer "<<slog::endl;
//        inferRequest.infer();

        //!< Do asynchronous inference
        size_t total_iterations = 4;
        size_t cur_iterations = 0;
        std::condition_variable condVar;//Used to wait and wakeup
        std::mutex mutex;//Mutex lock
        std::exception_ptr exceptionVar;//Used to scratch exception

        inferRequest.set_callback([&](std::exception_ptr ex) {
            std::lock_guard<std::mutex> lock(mutex);
            if (ex) {
                exceptionVar = ex;
                condVar.notify_all();
                return;
            }

            cur_iterations++;
            if (cur_iterations < total_iterations) {
                inferRequest.start_async();
                slog::info << "Comleted " << cur_iterations \
                << " asynchronous request execution" << slog::endl;
            } else {
                condVar.notify_one();
            }
        });

        inferRequest.start_async();//start first asynchronous

        std::unique_lock<std::mutex> uLock(mutex);
        condVar.wait(uLock, [&] {
            if (exceptionVar) {
                std::rethrow_exception(exceptionVar);
            }
            return cur_iterations == total_iterations;
        });

        slog::info << "Completed async requests execution" << slog::endl;

//!< Process output and Show result
        slog::info<<"Complete Infer"<<slog::endl;
        const ov::Tensor &outputTensor = inferRequest.get_output_tensor();
        using tensor_otype = ov::fundamental_type_for<ov::element::Type_t::f32>;//float here
        const tensor_otype *outputData = outputTensor.data<tensor_otype>();//float array here
        const int classSize = outputTensor.get_size() / batchSize;
        slog::info<<classSize<<slog::endl;
        std::vector<tensor_otype> resultConfidences;
        std::vector<int> result_indexs;
        for (int j = 0; j < batchSize; j++) {
            tensor_otype resultConfidence = 0;
            int result_index = 0;
            for (int i = 0; i < classSize; i++) {
                int index = i + j * classSize;
                if (outputData[index] > resultConfidence) {
                    resultConfidence = outputData[index];
                    result_index = i;
                }
            }
            resultConfidences.push_back(resultConfidence);
            result_indexs.push_back(result_index);
        }
        auto end = std::chrono::high_resolution_clock::now();
//        double time = ((double)cv::getTickCount() - t) / (double)cv::getTickFrequency();
        using duration_fps = std::chrono::duration<double>;
        for (size_t i = 0; i < resultConfidences.size(); i++) {
            slog::info << "Class:" << labeName[result_indexs[i]] \
            << "\t" << "Confidence:" << resultConfidences[i] << slog::endl;
        }

//        slog::info<<" inference time:"<<time<<slog::endl;
        slog::info<<" fps:"<< duration_fps(end-start).count()<<slog::endl;

    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }


    return 0;
}
