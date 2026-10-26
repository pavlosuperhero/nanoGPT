#pragma once

#include <string>
#include <filesystem>

struct AiConfig {
  int batch_size{64};
  int block_size{256};
  int max_iters{5000};
  int eval_interval{500};
  std::string learning_rate{"e3-4"};
  std::string device{"cuda"};
  int n_embd{384};
  int n_head{8};
  int n_layers{6};
  float dropout{0.2};
};

struct AiConfigFile {
  AiConfig config;

  static AiConfigFile load_from_file(const std::filesystem::path& path);
};
