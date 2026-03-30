#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "../models/models.h"

// Configuration remains for default hyper-parameters
struct TrainingConfig {
    int64_t img_w = 1024;
    int64_t img_h = 1024;
    int64_t channels = 3;     // <-- ADD THIS
    int64_t min_defect = 10;   // <-- ADD THIS
    int64_t batch_size = 4;
    int64_t epochs = 10;
    double learning_rate = 1e-4;
    torch::ScalarType precision = torch::kFloat32;
};

// --- Universal Functions ---

void draw_metrics(int epoch, int max_epochs, const std::vector<float>& losses, const std::vector<float>& accs) {
    int w = 600, h = 400;
    cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC3);
    
    // Draw background grid
    for(int i = 0; i <= 4; i++) {
        int y = i * (h / 4);
        cv::line(canvas, {0, y}, {w, y}, cv::Scalar(50, 50, 50), 1);
    }

    auto plot = [&](const std::vector<float>& data, cv::Scalar color, float max_val) {
        if (data.size() < 2) return;
        for (size_t i = 1; i < data.size(); i++) {
            cv::Point p1((i - 1) * w / max_epochs, h - (data[i - 1] / max_val * h));
            cv::Point p2(i * w / max_epochs, h - (data[i] / max_val * h));
            cv::line(canvas, p1, p2, color, 2);
        }
    };

    // Find max loss for scaling, accuracy is always 0.0-1.0
    float max_loss = *std::max_element(losses.begin(), losses.end());
    if (max_loss < 1.0f) max_loss = 1.0f;

    plot(losses, cv::Scalar(0, 0, 255), max_loss); // Red for Loss
    plot(accs, cv::Scalar(0, 255, 0), 1.0f);     // Green for Accuracy

    cv::putText(canvas, "Red: Loss (scaled)", {10, 20}, 1, 1, cv::Scalar(0, 0, 255));
    cv::putText(canvas, "Green: Accuracy", {10, 40}, 1, 1, cv::Scalar(0, 255, 0));
    
    cv::imshow("Training Metrics", canvas);
    cv::waitKey(1); // Refresh window
}

/**
 * Universal Training Function
 * Takes any model holder and any compatible data loaders.
 */
template <typename ModelType, typename DataLoader>
void train_universal(
    ModelType& model,
    DataLoader& train_loader,
    DataLoader& val_loader,
    torch::Device device,
    const TrainingConfig& config) {
    
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.learning_rate));
    float best_acc = 0.0;

    std::vector<float> epoch_losses;
    std::vector<float> epoch_accs;

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        // Training Phase
        model->train();
        float total_loss = 0;
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device).to(config.precision);
            auto targets = batch.target.to(device);

            optimizer.zero_grad();
            auto outputs = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(outputs.to(torch::kFloat32), targets);
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<float>();
            batch_count++;
        }

        // Calculate average loss for the epoch
        float avg_loss = total_loss / (batch_count > 0 ? batch_count : 1);
        epoch_losses.push_back(avg_loss);

        // Validation Phase
        model->eval();
        torch::NoGradGuard no_grad;
        int64_t tp = 0, tn = 0, fp = 0, fn = 0;

        for (auto& batch : *val_loader) {
            auto data = batch.data.to(device).to(config.precision);
            auto targets = batch.target.to(device);
            auto outputs = model->forward(data);
            auto preds = outputs.argmax(1);

            for (int i = 0; i < targets.size(0); ++i) {
                int64_t t = targets[i].item<int64_t>();
                int64_t p = preds[i].item<int64_t>();
                if (t == 1 && p == 1) tp++;
                else if (t == 0 && p == 0) tn++;
                else if (t == 0 && p == 1) fp++;
                else if (t == 1 && p == 0) fn++;
            }
        }

        float acc = (float)(tp + tn) / (tp + tn + fp + fn);
        epoch_accs.push_back(acc); // Track accuracy

        // Update the Graph
        draw_metrics(epoch, config.epochs, epoch_losses, epoch_accs);

        std::cout << "Epoch [" << epoch << "] Loss: " << avg_loss 
                  << " | Acc: " << acc * 100 << "% | Missed (FN): " << fn << std::endl;

        if (acc > best_acc) {
            best_acc = acc;
            torch::save(model, "best_model_universal.pt");
        }
    }
}

/**
 * Universal Inference Function
 * Takes a model, an image, and returns the top class index.
 */
template <typename ModelType>
int predict_universal(
    ModelType& model,
    cv::Mat& frame,
    int64_t img_w,
    int64_t img_h,
    torch::Device device,
    torch::ScalarType precision = torch::kFloat32) {

    model->eval();
    torch::NoGradGuard no_grad;

    // Standard OpenCV to Tensor conversion
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(img_w, img_h));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    torch::Tensor tensor = torch::from_blob(resized.data, {resized.rows, resized.cols, 3}, torch::kByte);
    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat).div(255.0).unsqueeze(0);
    tensor = tensor.to(device).to(precision);

    auto output = model->forward(tensor);
    return output.argmax(1).item<int>();
}

/**
 * Serialization Helpers
 */
template <typename ModelType>
void save_model(ModelType& model, const std::string& filename) {
    torch::save(model, filename);
}

template <typename ModelType>
void load_model(ModelType& model, const std::string& filename) {
    torch::load(model, filename);
}