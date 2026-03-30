#pragma once
#include <torch/torch.h>
#include <vector>

// 1. Define the Bottleneck Block (Standard for ResNet-50/101/152)
struct ResNetBottleneckImpl : torch::nn::Module {
    // Split comma-separated declarations and use = nullptr
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::Conv2d conv3 = nullptr;
    
    torch::nn::BatchNorm2d bn1 = nullptr;
    torch::nn::BatchNorm2d bn2 = nullptr;
    torch::nn::BatchNorm2d bn3 = nullptr;
    
    torch::nn::Sequential downsample = nullptr;

    ResNetBottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride = 1, torch::nn::Sequential downsample_mod = nullptr) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
        
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3).stride(stride).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));
        
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes * 4, 1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(planes * 4));
        
        if (downsample_mod) {
            this->downsample = register_module("downsample", downsample_mod);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;

        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = torch::relu(bn2->forward(conv2->forward(out)));
        out = bn3->forward(conv3->forward(out));

        if (downsample) identity = downsample->forward(x);

        out += identity; // Residual Connection
        return torch::relu(out);
    }
};
TORCH_MODULE(ResNetBottleneck);

// 2. Define the Main ResNet-50 Module
struct ResNet50_VisionImpl : torch::nn::Module {
    torch::nn::Sequential features = nullptr; // Changed from {nullptr}
    torch::nn::Linear classifier = nullptr;   // Changed from {nullptr}
    torch::ScalarType dtype;
    int64_t inplanes = 64;
    double lr; // Add this member

    ResNet50_VisionImpl(int64_t channels, 
                        int64_t img_w, 
                        int64_t img_h, 
                        int64_t min_defect = 10, 
                        int64_t num_classes = 2, 
                        torch::ScalarType precision = torch::kFloat32,
                        double learning_rate = 1e-4) // Add parameter here
        : dtype(precision), lr(learning_rate) { // Initialize here
        
        // --- Stem (Initial Convolution) ---
        features = register_module("features", torch::nn::Sequential());
        features->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 64, 7).stride(2).padding(3).bias(false)));
        features->push_back(torch::nn::BatchNorm2d(64));
        features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        features->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        // --- Standard ResNet-50 Layers (3, 4, 6, 3 blocks) ---
        _make_layer(64, 3, 1);
        _make_layer(128, 4, 2);
        _make_layer(256, 6, 2);
        _make_layer(512, 3, 2);

        // --- Global Average Pooling ---
        features->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));

        // --- Classifier (ResNet-50 expansion factor is 4, so 512 * 4 = 2048) ---
        classifier = register_module("classifier", torch::nn::Linear(2048, num_classes));

        this->to(dtype);
    }

    void _make_layer(int64_t planes, int64_t blocks, int64_t stride) {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes * 4) {
            downsample = torch::nn::Sequential();
            downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * 4, 1).stride(stride).bias(false)));
            downsample->push_back(torch::nn::BatchNorm2d(planes * 4));
        }

        features->push_back(ResNetBottleneck(inplanes, planes, stride, downsample));
        inplanes = planes * 4;
        for (int i = 1; i < blocks; i++) {
            features->push_back(ResNetBottleneck(inplanes, planes));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.to(dtype);
        x = features->forward(x);
        x = x.view({x.size(0), -1}); // Flatten 2048x1x1 to 2048
        return classifier->forward(x);
    }
};
TORCH_MODULE(ResNet50_Vision);