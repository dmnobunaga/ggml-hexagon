#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

struct llama_bench_battery_result {
    double mwh = -1.0;
    std::string source = "unavailable";
};

class llama_bench_battery_meter {
public:
    llama_bench_battery_meter() = default;
    llama_bench_battery_meter(const llama_bench_battery_meter &) = delete;
    llama_bench_battery_meter & operator=(const llama_bench_battery_meter &) = delete;

    ~llama_bench_battery_meter() {
        stop();
    }

    bool start() {
#if defined(__ANDROID__)
        const std::string state = read_text("/sys/class/power_supply/battery/status");
        if (state == "Charging" || state == "Full") {
            return false;
        }

        has_energy_  = read_value("/sys/class/power_supply/battery/energy_now", energy_start_);
        has_charge_  = read_value("/sys/class/power_supply/battery/charge_counter", charge_start_);
        has_voltage_ = read_value("/sys/class/power_supply/battery/voltage_now", voltage_start_);
        int64_t current = 0;
        has_current_ = has_voltage_ && read_value("/sys/class/power_supply/battery/current_now", current);

        if (!has_energy_ && !(has_charge_ && has_voltage_) && !has_current_) {
            return false;
        }
        if (has_current_) {
            stop_requested_ = false;
            worker_ = std::thread([this]() { sample_current(); });
        }
        started_ = true;
        return true;
#else
        return false;
#endif
    }

    llama_bench_battery_result finish() {
        llama_bench_battery_result result;
        if (!started_) {
            return result;
        }
        stop();
#if defined(__ANDROID__)
        const std::string state = read_text("/sys/class/power_supply/battery/status");
        if (state == "Charging" || state == "Full") {
            return result;
        }

        int64_t final_value = 0;
        if (has_energy_ && read_value("/sys/class/power_supply/battery/energy_now", final_value)) {
            const int64_t delta = energy_start_ - final_value;
            if (delta > 0) {
                result.mwh = double(delta) / 1000.0; // energy_now: uWh
                result.source = "battery_energy_now";
                return result;
            }
        }

        int64_t voltage_end = 0;
        if (has_charge_ && has_voltage_ &&
            read_value("/sys/class/power_supply/battery/charge_counter", final_value) &&
            read_value("/sys/class/power_supply/battery/voltage_now", voltage_end)) {
            const int64_t delta = charge_start_ - final_value;
            if (delta > 0 && voltage_start_ > 0 && voltage_end > 0) {
                const double avg_voltage_uv = (double(voltage_start_) + double(voltage_end)) / 2.0;
                result.mwh = double(delta) * avg_voltage_uv / 1e9; // uAh * uV -> mWh
                result.source = "battery_charge_counter";
                return result;
            }
        }

        if (has_current_ && current_samples_ >= 2 && integrated_mwh_ > 0) {
            result.mwh = integrated_mwh_;
            result.source = "battery_current_integral";
        }
#endif
        return result;
    }

private:
    bool started_ = false;
    bool has_energy_ = false;
    bool has_charge_ = false;
    bool has_voltage_ = false;
    bool has_current_ = false;
    int64_t energy_start_ = 0;
    int64_t charge_start_ = 0;
    int64_t voltage_start_ = 0;
    std::atomic<bool> stop_requested_{ false };
    std::thread worker_;
    double integrated_mwh_ = 0;
    int current_samples_ = 0;

    static bool read_value(const char * path, int64_t & value) {
        FILE * f = std::fopen(path, "r");
        if (!f) {
            return false;
        }
        long long v = 0;
        const bool ok = std::fscanf(f, "%lld", &v) == 1;
        std::fclose(f);
        if (ok) {
            value = int64_t(v);
        }
        return ok;
    }

    static std::string read_text(const char * path) {
        FILE * f = std::fopen(path, "r");
        if (!f) {
            return "";
        }
        char buf[64] = {};
        const bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
        std::fclose(f);
        if (!ok) {
            return "";
        }
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        return s;
    }

    void sample_current() {
        using clock = std::chrono::steady_clock;
        int64_t current = 0;
        int64_t voltage = 0;
        bool valid = read_value("/sys/class/power_supply/battery/current_now", current) &&
                     read_value("/sys/class/power_supply/battery/voltage_now", voltage);
        auto last = clock::now();
        double last_watts = valid ? std::abs(double(current) * double(voltage)) / 1e12 : 0.0;
        current_samples_ = valid ? 1 : 0;
        while (!stop_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (stop_requested_.load()) {
                break;
            }
            int64_t next_current = 0;
            int64_t next_voltage = 0;
            const bool next_valid =
                read_value("/sys/class/power_supply/battery/current_now", next_current) &&
                read_value("/sys/class/power_supply/battery/voltage_now", next_voltage);
            const auto now = clock::now();
            if (valid && next_valid && current_samples_ > 0) {
                const double watts = std::abs(double(next_current) * double(next_voltage)) / 1e12;
                const double seconds = std::chrono::duration<double>(now - last).count();
                integrated_mwh_ += (last_watts + watts) * 0.5 * seconds / 3.6;
                last_watts = watts;
            } else if (next_valid) {
                last_watts = std::abs(double(next_current) * double(next_voltage)) / 1e12;
            }
            valid = next_valid;
            last = now;
            if (next_valid) {
                current_samples_++;
            }
        }
    }

    void stop() {
        stop_requested_.store(true);
        if (worker_.joinable()) {
            worker_.join();
        }
        started_ = false;
    }
};
