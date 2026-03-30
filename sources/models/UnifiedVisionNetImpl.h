#pragma once
#include <torch/torch.h>
#include <vector>

struct UnifiedVisionNetImpl : torch::nn::Module {
    torch::nn::Sequential local_feat = nullptr;
    torch::nn::Sequential global_feat = nullptr;
    torch::nn::Linear classifier = nullptr;
    torch::ScalarType dtype;

    // Inside UnifiedVisionNetImpl.h
    UnifiedVisionNetImpl(int64_t channels, 
                        int64_t img_w, 
                        int64_t img_h, 
                        int64_t min_defect_size = 10, 
                        int64_t num_classes = 2,
                        torch::ScalarType precision = torch::kFloat32) 
                        : dtype(precision) {
    
    // 1. Register an EMPTY sequential first
    local_feat = register_module("local_feat", torch::nn::Sequential());
    
    int64_t curr_defect = min_defect_size;
    int64_t c_out = 16;
    int64_t c_in = channels;

    // 2. Push modules directly into the registered member
    while (curr_defect > 2) {
        local_feat->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, 3).padding(1)));
        local_feat->push_back(torch::nn::ReLU());
        local_feat->push_back(torch::nn::MaxPool2d(2));
        
        curr_defect /= 2;
        c_in = c_out;
        c_out *= 2;
    }

    // 3. Register global_feat and classifier normally
    global_feat = register_module("global_feat", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, 3).padding(1)), // Use c_in/c_out from loop
        torch::nn::ReLU(),
        torch::nn::AdaptiveAvgPool2d(4)
    ));

    int64_t flattened_dim = (c_in * 1 * 1) + (c_out * 4 * 4); 
    classifier = register_module("classifier", torch::nn::Linear(flattened_dim, num_classes));

    this->to(dtype);
}

    torch::Tensor forward(torch::Tensor x) {
        x = x.to(dtype);
        
        auto feat_map = local_feat->forward(x);
        
        // FIX: Extract the Tensor from the tuple returned by adaptive_max_pool2d
        // std::get<0> gets the 'values', std::get<1> would be 'indices'
        auto local_v_tuple = torch::adaptive_max_pool2d(feat_map, {1, 1});
        auto local_v = std::get<0>(local_v_tuple).flatten(1);
        
        auto global_v = global_feat->forward(feat_map).flatten(1);
        
        auto combined = torch::cat({local_v, global_v}, 1);
        return classifier->forward(combined);
    }
};
TORCH_MODULE(UnifiedVisionNet);