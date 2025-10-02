#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace callmaze {

template <typename T, typename F>
auto tap(T&& value, F&& func) -> T {
    func(value);
    return std::forward<T>(value);
}

template <typename T>
auto chain_value(T&& value) {
    return std::forward<T>(value);
}

template <typename T, typename F, typename... Fs>
auto chain_value(T&& value, F&& func, Fs&&... rest) {
    return chain_value(func(std::forward<T>(value)), std::forward<Fs>(rest)...);
}

class StageReport {
public:
    StageReport(std::string stage, double value)
        : stage_(std::move(stage)), value_(value) {}

    StageReport& annotate(std::string note) {
        notes_.push_back(std::move(note));
        return *this;
    }

    StageReport& tag(std::string key, std::string value) {
        tags_.emplace_back(std::move(key), std::move(value));
        return *this;
    }

    std::string summary() const {
        std::ostringstream oss;
        oss << "Stage: " << stage_ << " -> value: " << value_;
        if (!notes_.empty()) {
            oss << " | notes:";
            for (const auto& note : notes_) {
                oss << ' ' << note;
            }
        }
        if (!tags_.empty()) {
            oss << " | tags:";
            for (const auto& [key, value] : tags_) {
                oss << ' ' << key << '=' << value;
            }
        }
        return oss.str();
    }

    double value() const { return value_; }

private:
    std::string stage_;
    double value_;
    std::vector<std::string> notes_;
    std::vector<std::pair<std::string, std::string>> tags_;
};

class StageProxy;

class StageProxyOptions {
public:
    explicit StageProxyOptions(const StageProxy& owner) : owner_(owner) {}

    StageProxyOptions& smoothing(double alpha) {
        smoothing_ = alpha;
        return *this;
    }

    StageProxyOptions& bias(double b) {
        bias_ = b;
        return *this;
    }

    double finalize() const;

private:
    const StageProxy& owner_;
    double smoothing_{1.0};
    double bias_{0.0};
};

class StageProxy {
public:
    StageProxy() = default;
    explicit StageProxy(std::string name) : name_(std::move(name)) {}

    StageProxy& tune(double factor) {
        factor_ = factor;
        return *this;
    }

    StageProxy& limit(double value) {
        limit_ = value;
        return *this;
    }

    StageReport capture(double input) const {
        const double scaled = input * factor_;
        const double clamped = std::clamp(scaled, -limit_, limit_);
        StageReport report(name_, clamped);
        return report;
    }

    StageProxyOptions options() const { return StageProxyOptions(*this); }

    const std::string& name() const { return name_; }
    double factor() const { return factor_; }
    double limit() const { return limit_; }

private:
    std::string name_ = "unnamed";
    double factor_ = 1.0;
    double limit_ = 100.0;
};

inline double StageProxyOptions::finalize() const {
    const double base = owner_.factor();
    return base * smoothing_ + bias_ * owner_.limit() * 0.01;
}

class Logger {
public:
    Logger& info(const std::string& message) {
        buffer_ << "[info] " << message << '\n';
        return *this;
    }

    Logger& debug(const std::string& message) {
        buffer_ << "[debug] " << message << '\n';
        return *this;
    }

    Logger& warn(const std::string& message) {
        buffer_ << "[warn] " << message << '\n';
        return *this;
    }

    std::string str() const { return buffer_.str(); }

private:
    std::ostringstream buffer_;
};

template <typename T>
class RollingStatistic {
public:
    RollingStatistic& add(T value) {
        values_.push_back(value);
        return *this;
    }

    template <typename Reducer>
    T reduce(T seed, Reducer&& reducer) const {
        return std::accumulate(values_.begin(), values_.end(), seed,
                               std::forward<Reducer>(reducer));
    }

    T average() const {
        if (values_.empty()) {
            return T{};
        }
        const T sum = std::accumulate(values_.begin(), values_.end(), T{});
        return sum / static_cast<T>(values_.size());
    }

    std::size_t count() const { return values_.size(); }

private:
    std::vector<T> values_;
};

class Pipeline;

class PipelineCompound {
public:
    explicit PipelineCompound(Pipeline* owner) : owner_(owner) {}

    std::string describe() const;
    StageReport sample(double input) const;

private:
    Pipeline* owner_;
};

class Pipeline {
public:
    Pipeline() : compound(this) {}

    StageProxy& stage(const std::string& name) {
        auto [it, inserted] = stages_.try_emplace(name, StageProxy{name});
        (void)inserted;
        return it->second;
    }

    StageReport run(const std::string& name, double input) {
        return stage(name).capture(input);
    }

    std::vector<std::string> listing() const {
        std::vector<std::string> names;
        names.reserve(stages_.size());
        for (const auto& [name, _] : stages_) {
            names.push_back(name);
        }
        return names;
    }

    PipelineCompound compound;

private:
    std::map<std::string, StageProxy> stages_;
};

inline std::string PipelineCompound::describe() const {
    if (!owner_) {
        return "<no pipeline>";
    }
    auto names = owner_->listing();
    if (names.empty()) {
        return "<empty pipeline>";
    }
    std::ostringstream oss;
    oss << "Pipeline stages:";
    for (const auto& name : names) {
        oss << ' ' << name;
    }
    return oss.str();
}

inline StageReport PipelineCompound::sample(double input) const {
    if (!owner_) {
        return StageReport("unavailable", input);
    }
    auto names = owner_->listing();
    if (names.empty()) {
        return StageReport("undefined", input);
    }
    return owner_->run(names.front(), input);
}

class Analyzer {
public:
    Analyzer(Logger& logger, Pipeline& pipeline)
        : logger_(&logger), pipeline_(&pipeline) {}

    Analyzer& enable(bool value = true) {
        enabled_ = value;
        logger_->debug(value ? "Analyzer enabled" : "Analyzer disabled");
        return *this;
    }

    Analyzer& setThreshold(double threshold) {
        threshold_ = threshold;
        return *this;
    }

    StageReport analyze(const std::vector<double>& values) {
        if (!enabled_) {
            logger_->warn("Analyzer is disabled; returning stub result");
            return StageReport("disabled", 0.0);
        }
        RollingStatistic<double> stats;
        for (double value : values) {
            stats.add(value);
        }
        const double average = stats.average();
        const double adjusted = average - threshold_;
        auto& primary = pipeline_->stage("primary");
        auto report = primary.capture(adjusted);
        report.annotate("threshold compensated");
        report.tag("count", std::to_string(stats.count()));
        const double optionScore = primary.options().smoothing(0.8).bias(0.5).finalize();
        report.tag("option", std::to_string(optionScore));
        logger_->info("Analyzer produced value " + std::to_string(report.value()));
        return report;
    }

private:
    Logger* logger_;
    Pipeline* pipeline_;
    bool enabled_ = false;
    double threshold_ = 0.0;
};

struct EngineDashboard {
    struct LogView {
        Logger* logger = nullptr;

        std::string print() const { return logger ? logger->str() : std::string{}; }
    };

    StageReport latestReport{"bootstrap", 0.0};
    LogView logView;
};

class Engine {
public:
    Engine()
        : pipeline_(), analyzer_(logger_, pipeline_), dashboard_{} {
        dashboard_.logView.logger = &logger_;
    }

    Pipeline& pipeline() { return pipeline_; }
    Pipeline* accessor() { return &pipeline_; }
    Analyzer& analyzer() { return analyzer_; }
    EngineDashboard& dashboard() { return dashboard_; }

    Engine& calibrate(std::function<double(double)> fn) {
        calibration_ = std::move(fn);
        return *this;
    }

    double process(const std::vector<double>& input) {
        auto normalized = input;
        if (calibration_) {
            std::transform(normalized.begin(), normalized.end(), normalized.begin(), calibration_);
        }
        auto report = analyzer_.analyze(normalized);
        dashboard_.latestReport = report;
        logger_.debug(report.summary());
        return report.value();
    }

    std::string flushLog() const { return logger_.str(); }

private:
    Logger logger_;
    Pipeline pipeline_;
    Analyzer analyzer_;
    EngineDashboard dashboard_;
    std::function<double(double)> calibration_;
};

}  // namespace callmaze

