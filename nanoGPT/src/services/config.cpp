#include "config.hpp"
#include <stdexcept>
#include <yaml-cpp/yaml.h>

AiConfigFile AiConfigFile::load_from_file(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Config file not found: " + path.string());
  }

  YAML::Node root = YAML::LoadFile(path.string());
  AiConfigFile config;

  if (root["LEARN"]) {
    auto learn_config = root["LEARN"];
    config.config.learning_rate = learn_config["LEARNING_RATE"].as<std::string>(config.config.learning_rate);
    config.config.device = learn_config["DEVICE"].as<std::string>(config.config.device);
    config.config.batch_size = learn_config["BATCH_SIZE"].as<int>(config.config.batch_size);
    config.config.block_size = learn_config["BLOCK_SIZE"].as<int>(config.config.block_size);
    config.config.max_iters = learn_config["MAX_ITERS"].as<int>(config.config.max_iters);
    config.config.eval_interval = learn_config["EVAL_INTERVAL"].as<int>(config.config.eval_interval);
    config.config.n_embd = learn_config["N_EMBD"].as<int>(config.config.n_embd);
    config.config.n_head = learn_config["N_HEAD"].as<int>(config.config.n_head);
    config.config.n_layers = learn_config["N_LAYERS"].as<int>(config.config.n_layers);
    config.config.dropout = learn_config["DROPOUT"].as<float>(config.config.dropout);
    } else {
    throw std::runtime_error("Missing LEARNING configurations!!");
    }
  return config;
}
