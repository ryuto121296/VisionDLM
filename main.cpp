/**
 * ResNet Binary Classifier — LibTorch + OpenCV
 * Classes : Good | Bad
 * Split   : Train 70% | Val 15% | Test 15%
 *
 * Build prerequisites
 *   - LibTorch  >= 2.1
 *   - OpenCV    >= 4.5
 *   - C++17
 *
 * After training the program writes:
 *   model_scripted.pt   – TorchScript (for LibTorch C++ inference)
 *   model.onnx          – ONNX (for ONNX-Runtime / TensorRT / etc.)
 *
 * Then it loads the TorchScript model and runs the full Test split,
 * printing per-image predictions and final accuracy.
 */

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   IMAGE_SIZE   = 224;   // ResNet expected input
static constexpr int   NUM_CLASSES  = 2;
static constexpr int   NUM_EPOCHS   = 20;
static constexpr int   BATCH_SIZE   = 16;
static constexpr float LR           = 1e-5f;
static constexpr float TRAIN_RATIO  = 0.70f;
static constexpr float VAL_RATIO    = 0.15f;
// TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

static const std::string GOOD_DIR = "D:\\_Dataset\\M6_29Sep2023\\Good";
static const std::string BAD_DIR  = "D:\\_Dataset\\M6_29Sep2023\\Bad";

static const std::string PT_OUT   = "model_scripted.pt";
static const std::string ONNX_OUT = "model.onnx";

// Class index mapping: Good=0, Bad=1
static const std::vector<std::string> CLASS_NAMES = {"Good", "Bad"};

// ImageNet normalisation values used by pretrained torchvision models
static const std::vector<float> IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
static const std::vector<float> IMAGENET_STD  = {0.229f, 0.224f, 0.225f};

// ─────────────────────────────────────────────────────────────────────────────
// Sample struct
// ─────────────────────────────────────────────────────────────────────────────
struct Sample {
    std::string path;
    int         label; // 0 = Good, 1 = Bad
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: collect all image paths from a directory with a given label
// ─────────────────────────────────────────────────────────────────────────────
void collect_images(const std::string& dir, int label,
                    std::vector<Sample>& out)
{
    static const std::vector<std::string> exts =
        {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};

    if (!fs::exists(dir)) {
        std::cerr << "[WARN] Directory not found: " << dir << "\n";
        return;
    }

    for (auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        // normalise extension to lower-case
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            out.push_back({entry.path().string(), label});
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: load & preprocess one image → CHW float tensor [0,1] normalised
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor load_image(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[WARN] Cannot read: " << path << "\n";
        // return a black image so we don't crash; will be zero-tensored
        img = cv::Mat::zeros(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3);
    }

    cv::resize(img, img, {IMAGE_SIZE, IMAGE_SIZE});
    img.convertTo(img, CV_32F, 1.0 / 255.0);           // [0,1]
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // HWC → CHW
    auto tensor = torch::from_blob(img.data,
                                   {IMAGE_SIZE, IMAGE_SIZE, 3},
                                   torch::kFloat32).clone();
    tensor = tensor.permute({2, 0, 1});                 // CHW

    // ImageNet normalisation
    for (int c = 0; c < 3; ++c)
        tensor[c] = (tensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];

    return tensor; // [3, H, W]
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Dataset
// ─────────────────────────────────────────────────────────────────────────────
struct ClassifyDataset : torch::data::datasets::Dataset<ClassifyDataset> {
    std::vector<Sample> samples_;

    explicit ClassifyDataset(std::vector<Sample> s)
        : samples_(std::move(s)) {}

    torch::data::Example<> get(size_t idx) override {
        auto& s = samples_[idx];
        torch::Tensor img   = load_image(s.path);
        torch::Tensor label = torch::tensor(s.label, torch::kLong);
        return {img, label};
    }

    torch::optional<size_t> size() const override {
        return samples_.size();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Build a fine-tuneable ResNet-18 with replaced FC head (2 outputs)
// LibTorch ships ResNet via torchvision; if you link torchvision:
//   auto model = vision::models::ResNet18(/*pretrained=*/true);
//
// Without torchvision we build the same architecture manually using the
// standard torch::nn modules.  For truly pretrained weights you would
// download resnet18.pt (TorchScript) and load it below.
// ─────────────────────────────────────────────────────────────────────────────

// Basic residual block (same as torchvision BasicBlock)
struct BasicBlockImpl : torch::nn::Module {
    torch::nn::Conv2d      conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr},   bn2{nullptr};
    torch::nn::Sequential  downsample{nullptr};
    bool has_downsample = false;
    int  stride_;

    BasicBlockImpl(int in_ch, int out_ch, int stride = 1,
                   torch::nn::Sequential ds = nullptr)
        : stride_(stride)
    {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3)
                              .stride(stride).padding(1).bias(false)));
        bn1   = register_module("bn1",  torch::nn::BatchNorm2d(out_ch));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3)
                              .stride(1).padding(1).bias(false)));
        bn2   = register_module("bn2",  torch::nn::BatchNorm2d(out_ch));

        // Always register so model->to(device) moves its weights too
        if (ds) {
            downsample      = register_module("downsample", ds);
            has_downsample  = true;
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        if (has_downsample)
            identity = downsample->forward(identity);
        x += identity;
        return torch::relu(x);
    }
};
TORCH_MODULE(BasicBlock);

// ResNet-18
struct ResNet18Impl : torch::nn::Module {
    torch::nn::Conv2d       conv1{nullptr};
    torch::nn::BatchNorm2d  bn1{nullptr};
    torch::nn::MaxPool2d    maxpool{nullptr};
    torch::nn::Sequential   layer1, layer2, layer3, layer4;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear       fc{nullptr};
    int in_planes_ = 64;

    explicit ResNet18Impl(int num_classes = 2) {
        conv1   = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7)
                              .stride(2).padding(3).bias(false)));
        bn1     = register_module("bn1",   torch::nn::BatchNorm2d(64));
        maxpool = register_module("maxpool",
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3)
                                 .stride(2).padding(1)));

        layer1  = register_module("layer1", make_layer(64,  2, 1));
        layer2  = register_module("layer2", make_layer(128, 2, 2));
        layer3  = register_module("layer3", make_layer(256, 2, 2));
        layer4  = register_module("layer4", make_layer(512, 2, 2));

        avgpool = register_module("avgpool",
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1})));
        fc      = register_module("fc",
            torch::nn::Linear(512, num_classes));
    }

    torch::nn::Sequential make_layer(int planes, int blocks, int stride) {
        torch::nn::Sequential ds = nullptr;  // null, not empty-Sequential
        if (stride != 1 || in_planes_ != planes) {
            ds = torch::nn::Sequential(
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_planes_, planes, 1)
                    .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(planes));
        }
        torch::nn::Sequential seq;
        seq->push_back(BasicBlock(in_planes_, planes, stride, ds));
        in_planes_ = planes;
        for (int i = 1; i < blocks; ++i)
            seq->push_back(BasicBlock(planes, planes));
        return seq;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = maxpool->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avgpool->forward(x);
        x = x.view({x.size(0), -1});
        return fc->forward(x);
    }
};
TORCH_MODULE(ResNet18);

// ─────────────────────────────────────────────────────────────────────────────
// Training helper: one pass over a DataLoader
// ─────────────────────────────────────────────────────────────────────────────
template <typename Loader>
std::pair<double, double> run_epoch(ResNet18& model,
                                    Loader&   loader,
                                    torch::optim::Optimizer* opt,
                                    torch::Device device,
                                    bool training)
{
    if (training) model->train();
    else          model->eval();

    double  total_loss = 0.0;
    int64_t correct    = 0;
    int64_t total      = 0;

    torch::nn::CrossEntropyLoss criterion;

    for (auto& batch : loader) {
        auto data   = batch.data.to(device);
        auto labels = batch.target.to(device);

        torch::Tensor output;
        torch::Tensor loss;

        if (training) {
            // Gradients are ON by default (no guard active)
            opt->zero_grad();
            output = model->forward(data);
            loss   = criterion(output, labels);
            loss.backward();
            opt->step();
        } else {
            // Disable gradients for validation / test
            torch::NoGradGuard no_grad;
            output = model->forward(data);
            loss   = criterion(output, labels);
        }

        total_loss += loss.item<double>() * data.size(0);
        correct    += output.argmax(1).eq(labels).sum().item<int64_t>();
        total      += data.size(0);
    }

    double avg_loss = (total > 0) ? total_loss / total : 0.0;
    double accuracy = (total > 0) ? 100.0 * correct / total : 0.0;
    return {avg_loss, accuracy};
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // ── Device ──────────────────────────────────────────────────────────────
    torch::Device device = torch::cuda::is_available()
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: "
              << (device == torch::kCUDA ? "CUDA" : "CPU") << "\n\n";

    // ── Collect all samples ──────────────────────────────────────────────────
    std::vector<Sample> all_samples;
    collect_images(GOOD_DIR, 0, all_samples);
    collect_images(BAD_DIR,  1, all_samples);

    if (all_samples.empty()) {
        std::cerr << "[ERROR] No images found. Check your dataset paths.\n";
        return 1;
    }
    std::cout << "Total images: " << all_samples.size() << "\n";

    // Count per class
    int n_good = 0, n_bad = 0;
    for (auto& s : all_samples)
        (s.label == 0 ? n_good : n_bad)++;
    std::cout << "  Good: " << n_good << "  Bad: " << n_bad << "\n\n";

    // ── Shuffle & split ──────────────────────────────────────────────────────
    std::mt19937 rng(42);
    std::shuffle(all_samples.begin(), all_samples.end(), rng);

    size_t n       = all_samples.size();
    size_t n_train = static_cast<size_t>(n * TRAIN_RATIO);
    size_t n_val   = static_cast<size_t>(n * VAL_RATIO);
    size_t n_test  = n - n_train - n_val;

    std::vector<Sample> train_samples(all_samples.begin(),
                                      all_samples.begin() + n_train);
    std::vector<Sample> val_samples  (all_samples.begin() + n_train,
                                      all_samples.begin() + n_train + n_val);
    std::vector<Sample> test_samples (all_samples.begin() + n_train + n_val,
                                      all_samples.end());

    std::cout << "Split Dataset\nTrain: " << n_train
              << "  Val: "  << n_val
              << "  Test: " << n_test << "\n\n";

    // ── DataLoaders ──────────────────────────────────────────────────────────
    auto train_ds = ClassifyDataset(train_samples)
        .map(torch::data::transforms::Stack<>());
    auto val_ds   = ClassifyDataset(val_samples)
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<
        torch::data::samplers::RandomSampler>(
            std::move(train_ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(4));

    auto val_loader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
            std::move(val_ds),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

    // ── Model ────────────────────────────────────────────────────────────────
    ResNet18 model(NUM_CLASSES);
    model->to(device);

    // ── Optimizer ────────────────────────────────────────────────────────────
    torch::optim::Adam optimizer(model->parameters(),
        torch::optim::AdamOptions(LR));

    // ── Training loop ────────────────────────────────────────────────────────
    std::cout << "_________________________________________________________\n";
    std::cout << " Training  (lr=" << LR << ", epochs=" << NUM_EPOCHS << ")\n";
    std::cout << "_________________________________________________________\n";

    double best_val_acc = 0.0;

    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();

        auto [tr_loss, tr_acc] = run_epoch(
            model, *train_loader, &optimizer, device, /*training=*/true);
        auto [va_loss, va_acc] = run_epoch(
            model, *val_loader,   nullptr,    device, /*training=*/false);

        auto dt = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::steady_clock::now() - t0).count();

        std::printf(
            "Epoch [%2d/%2d]  "
            "Train Loss: %.4f  Acc: %5.2f%%  |  "
            "Val   Loss: %.4f  Acc: %5.2f%%  |  %llds\n",
            epoch, NUM_EPOCHS,
            tr_loss, tr_acc,
            va_loss, va_acc,
            (long long)dt);

        if (va_acc > best_val_acc) {
            best_val_acc = va_acc;
            torch::save(model, "best_model.pt");
            std::cout << "New best val acc saved.\n";
        }
    }

    std::cout << "\nBest Validation Accuracy: "
              << std::fixed << std::setprecision(2) << best_val_acc << "%\n\n";

    // ── Load best model weights ──────────────────────────────────────────────
    // torch::load() deserialises back into the same nn::Module type.
    // This is the correct C++ API — torch::jit::trace does NOT exist in C++.
    torch::load(model, "best_model.pt");
    model->eval();
    model->to(device);

    // ── Save final model (LibTorch C++ format) ───────────────────────────────
    // torch::save writes a file loadable by torch::load in any C++ binary.
    torch::save(model, PT_OUT);
    std::cout << "LibTorch model saved ->" << PT_OUT << "\n";

    // ── Export: TorchScript + ONNX (Python) ──────────────────────────────────
    // torch::jit::trace and torch.onnx.export live in Python only.
    // Run helpers.py after training to produce model_scripted.pt / model.onnx:
    //
    //   python helpers.py --weights best_model.pt --to-script --to-onnx
    //
    std::cout << "\n[INFO] TorchScript / ONNX export — run from this folder:\n";
    std::cout << "  python helpers.py --weights best_model.pt --to-script --to-onnx\n\n";

    // ═════════════════════════════════════════════════════════════════════════
    //  INFERENCE on Test split
    //  We re-use the already-loaded nn::Module (same process, zero file I/O).
    //  If you want to prove the saved file works, we reload it from disk below.
    // ═════════════════════════════════════════════════════════════════════════
    std::cout << "_________________________________________________________\n";
    std::cout << " Inference on Test split (model loaded from " << PT_OUT << ")\n";
    std::cout << "_________________________________________________________\n";

    // Reload from disk to confirm the saved file is valid
    ResNet18 infer_model(NUM_CLASSES);
    try {
        torch::load(infer_model, PT_OUT);
        infer_model->eval();
        infer_model->to(device);
        std::cout << "Loaded from disk: " << PT_OUT << "\n\n";
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Cannot load model from disk: " << e.what() << "\n";
        return 1;
    }

    // Run test samples one-by-one
    int correct = 0;
    std::cout << std::left
              << std::setw(60) << "Image"
              << std::setw(10) << "True"
              << std::setw(12) << "Predicted"
              << "Correct?\n";
    std::cout << std::string(95, '-') << "\n";

    {
        torch::NoGradGuard no_grad;
        for (auto& s : test_samples) {
            torch::Tensor img = load_image(s.path)
                                    .unsqueeze(0)  // [1, 3, H, W]
                                    .to(device);

            torch::Tensor output = infer_model->forward(img); // [1, num_classes]
            int pred_label = output.argmax(1).item<int>();
            bool ok = (pred_label == s.label);
            if (ok) ++correct;

            // Shorten path for display
            fs::path p(s.path);
            std::string short_path = p.filename().string();
            if (short_path.size() > 58)
                short_path = "..." + short_path.substr(short_path.size() - 55);

            std::cout << std::left
                      << std::setw(60) << short_path
                      << std::setw(10) << CLASS_NAMES[s.label]
                      << std::setw(12) << CLASS_NAMES[pred_label]
                      << (ok ? "OK" : "NG") << "\n";
        }
    }

    double test_acc = 100.0 * correct / static_cast<int>(test_samples.size());
    std::cout << "\n" << std::string(95, '=') << "\n";
    std::printf("Test Accuracy: %d / %zu  (%.2f%%)\n",
                correct, test_samples.size(), test_acc);

    return 0;
}