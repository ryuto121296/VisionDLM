#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "sources/models/models.h"
#include "sources/algorithm/model_handler.h"
#include "sources/algorithm/dataset.h"

namespace fs = std::filesystem;

int main() {
    // 1. Setup Configuration
    TrainingConfig config;
    config.img_w = 1024;
    config.img_h = 1024;
    config.min_defect = 10;
    config.batch_size = 2; // Keep small for 1024x1024 to avoid OOM
    config.epochs = 20;
    config.precision = torch::kFloat32;

    // 2. Define Dataset Paths and Classes
    std::vector<std::string> class_list = {"Good", "Bad"};
    std::map<std::string, std::string> paths = {
        {"Good", "D:\\_Dataset\\M6_29Sep2023\\Good"},
        {"Bad", "D:\\_Dataset\\M6_29Sep2023\\Bad"}
    };

    std::vector<cv::Mat> all_imgs;
    std::vector<std::string> all_labels;

    // 3. Load Images from Disk
    std::cout << "Loading dataset..." << std::endl;
    for (const auto& [label, path] : paths) {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                cv::Mat img = cv::imread(entry.path().string());
                if (!img.empty()) {
                    all_imgs.push_back(img);
                    all_labels.push_back(label);
                }
            }
        }
    }
    std::cout << "Loaded " << all_imgs.size() << " images." << std::endl;

    // 4. Simple Train/Val Split (e.g., 80/20)
    size_t split_idx = static_cast<size_t>(all_imgs.size() * 0.8);
    
    std::vector<cv::Mat> train_imgs(all_imgs.begin(), all_imgs.begin() + split_idx);
    std::vector<std::string> train_labels(all_labels.begin(), all_labels.begin() + split_idx);
    
    std::vector<cv::Mat> val_imgs(all_imgs.begin() + split_idx, all_imgs.end());
    std::vector<std::string> val_labels(all_labels.begin() + split_idx, all_labels.end());

    // 5. Initialize Model and Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    auto model = UnifiedVisionNet(config.channels, config.img_w, config.img_h, config.min_defect, class_list.size(), config.precision);
    model->to(device);

    // 6. Create Data Loaders
    auto train_set = VisionDataset(train_imgs, train_labels, class_list, config.img_w, config.img_h)
                        .map(torch::data::transforms::Stack<>());
    auto val_set = VisionDataset(val_imgs, val_labels, class_list, config.img_w, config.img_h)
                        .map(torch::data::transforms::Stack<>());

    // Use simple make_data_loader; LibTorch will use RandomSampler by default for training 
    // and SequentialSampler if you provide specific options.
    auto train_loader = torch::data::make_data_loader(
        std::move(train_set), 
        torch::data::DataLoaderOptions().batch_size(config.batch_size).workers(2));

    auto val_loader = torch::data::make_data_loader(
        std::move(val_set), 
        torch::data::DataLoaderOptions().batch_size(config.batch_size));

    // 7. Train and Export
    std::cout << "Starting Training..." << std::endl;
    train_universal(model, train_loader, val_loader, device, config);
    save_model(model, "m6_vision_model.pt");

    // 8. Test Inference on a single image
    if (!val_imgs.empty()) {
        int result_idx = predict_universal(model, val_imgs[0], config.img_w, config.img_h, device, config.precision);
        std::cout << "Test Inference Result: " << class_list[result_idx] << std::endl;
    }

    return 0;
}