#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class VisionDataset : public torch::data::Dataset<VisionDataset> {
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    int width, height;

public:
    VisionDataset(const std::vector<cv::Mat>& imgs, const std::vector<std::string>& lbls, 
                  const std::vector<std::string>& classes, int w, int h) 
        : images(imgs), width(w), height(h) {
        for (const auto& l : lbls) {
            auto it = std::find(classes.begin(), classes.end(), l);
            labels.push_back(std::distance(classes.begin(), it));
        }
    }

    torch::data::Example<> get(size_t index) override {
        cv::Mat resized;
        cv::resize(images[index], resized, cv::Size(width, height));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        torch::Tensor t = torch::from_blob(resized.data, {height, width, 3}, torch::kByte).clone();
        t = t.permute({2, 0, 1}).to(torch::kFloat).div(255.0);
        return {t, torch::tensor(labels[index], torch::kLong)};
    }

    torch::optional<size_t> size() const override { return images.size(); }
};