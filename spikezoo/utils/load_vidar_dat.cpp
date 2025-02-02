#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <string>
#include <iostream>
#include <cstring>

namespace py = pybind11;

// 预计算查找表
static const std::array<std::array<float, 8>, 256> LOOKUP_TABLE = []() {
    std::array<std::array<float, 8>, 256> table{};
    for (int i = 0; i < 256; ++i) {
        for (int bit = 0; bit < 8; ++bit) {
            table[i][bit] = (i & (1 << bit)) ? 1.0f : 0.0f;
        }
    }
    return table;
}();

std::vector<uint8_t> load_files(const std::vector<std::string>& filenames) {
    std::vector<uint8_t> combined_data;
    for (const auto& filename : filenames) {
        // 打开文件
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);

        // 获取文件大小并读取数据
        size_t size = file.tellg();
        file.seekg(0);
        combined_data.resize(combined_data.size() + size);
        if (!file.read(reinterpret_cast<char*>(&combined_data[combined_data.size() - size]), size))
            throw std::runtime_error("Read failed: " + filename);
    }
    return combined_data;
}

py::array_t<float> load_vidar_dat_cpp(const py::object& input, int height = 250, int width = 400) {
    std::vector<std::string> filenames;

    // 检查输入类型
    if (py::isinstance<py::str>(input)) {
        filenames = {py::str(input).cast<std::string>()};
    } else if (py::isinstance<py::list>(input) || py::isinstance<py::tuple>(input)) {
        for (const auto& item : input) {
            if (!py::isinstance<py::str>(item)) {
                throw std::runtime_error("Input list contains non-string elements");
            }
            filenames.push_back(item.cast<std::string>());
        }
    } else {
        throw std::runtime_error("Input must be a string or list/tuple of strings");
    }

    // 加载文件
    std::vector<uint8_t> buffer = load_files(filenames);

    const int len_per_frame = height * width / 8;
    const int img_size = height * width;
    const int total_frames = buffer.size() / len_per_frame;

    // 创建输出numpy数组
    py::array_t<float> result({total_frames, height, width});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);

    auto spike_data = buffer.data();

    // 多线程处理逻辑
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = std::max(1u, std::min(num_threads, static_cast<unsigned int>(total_frames / 10)));
    std::vector<std::thread> threads;

    // 定义处理lambda
    auto process_frames = [&](size_t start_frame, size_t end_frame) {
        for (size_t frame_idx = start_frame; frame_idx < end_frame; ++frame_idx) {
            for (size_t pel = 0; pel < static_cast<size_t>(len_per_frame); ++pel) {
                uint8_t spike = spike_data[frame_idx * len_per_frame + pel];

                // 直接从预计算表中复制8个浮点数值
                std::memcpy(
                    ptr + frame_idx * img_size + (pel * 8),
                    LOOKUP_TABLE[spike].data(),
                    8 * sizeof(float)
                );
            }

            // 上下翻转
            for (size_t i = 0; i < height / 2; ++i) {
                size_t row1 = frame_idx * img_size + i * width;
                size_t row2 = frame_idx * img_size + (height - i - 1) * width;
                std::swap_ranges(ptr + row1, ptr + row1 + width, ptr + row2);
            }
        }
    };

    // 任务分割逻辑
    size_t chunk = total_frames / num_threads;
    size_t remainder = total_frames % num_threads;
    size_t start = 0;

    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t end = start + chunk + (i < remainder ? 1 : 0);
        threads.emplace_back(process_frames, start, end);
        start = end;
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    return result;
}

PYBIND11_MODULE(vidar_loader, m) {
    m.def("load_vidar_dat_cpp", &load_vidar_dat_cpp,
          py::arg("input"),
          py::arg("height") = 250,
          py::arg("width") = 400);
}