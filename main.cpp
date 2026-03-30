#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "sources/models/models.h"
#include "sources/algorithm/model_handler.h"
#include "sources/algorithm/dataset.h"

namespace fs = std::filesystem;

int main() {
    //check if CUDA worrks
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }
    // 1. Setup Configuration
    // Ensure TrainingConfig in model_handler.h has 'channels' and 'min_defect' members
    TrainingConfig config;
    config.img_w = 431;
    config.img_h = 431;
    config.channels = 1;    // Ensure this is set for RGB
    config.min_defect = 10; 
    config.batch_size = 1;  // NOTE: ResNet-50 is heavy. Use batch_size 1 for 1024x1024 to avoid OOM.
    config.epochs = 50;
    config.precision = torch::kFloat32;
    config.learning_rate = 1e-6;// Pass the learning rate from your config

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
        if (!fs::exists(path)) continue;
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

    if (all_imgs.empty()) return -1;

    // 4. Simple Train/Val Split (80/20)
    size_t split_idx = static_cast<size_t>(all_imgs.size() * 0.8);
    
    std::vector<cv::Mat> train_imgs(all_imgs.begin(), all_imgs.begin() + split_idx);
    std::vector<std::string> train_labels(all_labels.begin(), all_labels.begin() + split_idx);
    
    std::vector<cv::Mat> val_imgs(all_imgs.begin() + split_idx, all_imgs.end());
    std::vector<std::string> val_labels(all_labels.begin() + split_idx, all_labels.end());

    // 5. Initialize ResNet-50 Model and Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    
    // CHANGE: Use ResNet50_Vision instead of UnifiedVisionNet
    auto model = ResNet50_Vision(
        config.channels, 
        config.img_w, 
        config.img_h, 
        config.min_defect, 
        class_list.size(), 
        config.precision,
        config.learning_rate // Added parameter
    );
    
    model->to(device);

    // 6. Create Data Loaders
    auto train_set = VisionDataset(train_imgs, train_labels, class_list, config.img_w, config.img_h)
                        .map(torch::data::transforms::Stack<>());
    auto val_set = VisionDataset(val_imgs, val_labels, class_list, config.img_w, config.img_h)
                        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_set), 
        torch::data::DataLoaderOptions().batch_size(config.batch_size).workers(2));

    auto val_loader = torch::data::make_data_loader(
        std::move(val_set), 
        torch::data::DataLoaderOptions().batch_size(config.batch_size));

    // 7. Train and Export
    std::cout << "Starting ResNet-50 Training..." << std::endl;
    train_universal(model, train_loader, val_loader, device, config);
    save_model(model, "resnet50_vision_model.pt");

    // 8. Test Inference
    if (!val_imgs.empty()) {
        int result_idx = predict_universal(model, val_imgs[0], config.img_w, config.img_h, device, config.precision);
        std::cout << "Test Inference Result: " << class_list[result_idx] << std::endl;
    }

    return 0;
}