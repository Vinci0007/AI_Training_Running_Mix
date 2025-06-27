// #include "lib_import.hpp"

#if defined(_WIN32) || defined(_WIN64)
#include "torch/torch.h"
#endif

int main() {
    std::cout << "Hello, torch_build!" << std::endl;
    if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
    } else {
        std::cout << "CUDA is not available." << std::endl;
    }
    return 0;
}