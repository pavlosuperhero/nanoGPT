#include "services/config.hpp"
#include <iostream>
#include <string_view>
#include <filesystem>
#include <yaml-cpp/yaml.h>

void print_usage(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  -c, --config <path>  Path to YAML configuration file (default: config.yaml)\n"
              << "  -h, --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::filesystem::path config_path = "config.yaml";

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config_path = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a file path argument.\n\n";
                print_usage(argv[0]);
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    try {
        AiConfigFile cfg = AiConfigFile::load_from_file("config.yaml");
        const auto& aiconfig = cfg.config;

//       std::cout << "=== Model Configuration ===\n"
//                 << "batch_size:    " << aiconfig.batch_size << "\n"
//                 << "block_size:    " << aiconfig.block_size << "\n"
//                 << "max_iters:     " << aiconfig.max_iters << "\n"
//                 << "eval_interval: " << aiconfig.eval_interval << "\n"
//                 << "learning_rate: " << aiconfig.learning_rate << "\n"
//                 << "device:        " << aiconfig.device << "\n"
//                 << "n_embd:        " << aiconfig.n_embd << "\n"
//                 << "n_head:        " << aiconfig.n_head << "\n"
//                 << "n_layers:      " << aiconfig.n_layers << "\n"
//                 << "dropout:       " << aiconfig.dropout << "\n"
//                 << "===========================\n";

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML syntax error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
