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

// Indexable container classes for testing indexing functionality
template <typename T>
class DataArray {
public:
    DataArray() = default;
    explicit DataArray(std::vector<T> data) : data_(std::move(data)) {}

    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    size_t size() const { return data_.size(); }

    T& get(size_t index) { return data_[index]; }
    const T& get(size_t index) const { return data_[index]; }

private:
    std::vector<T> data_;
};

template <typename K, typename V>
class DataMap {
public:
    DataMap() = default;
    explicit DataMap(std::map<K, V> data) : data_(std::move(data)) {}

    V& operator[](const K& key) { return data_[key]; }
    const V& operator[](const K& key) const { return data_.at(key); }

    V& get(const K& key) { return data_[key]; }
    const V& get(const K& key) const { return data_.at(key); }

private:
    std::map<K, V> data_;
};

class DataProcessor {
public:
    DataProcessor() = default;

    DataProcessor& process() {
        processed_ = true;
        return *this;
    }

    bool isProcessed() const { return processed_; }

private:
    bool processed_ = false;
};

class Sensor {
public:
    Sensor() = default;

    Sensor& calibrate() {
        calibrated_ = true;
        return *this;
    }

    bool isCalibrated() const { return calibrated_; }

private:
    bool calibrated_ = false;
};

class ConfigValue {
public:
    ConfigValue() = default;
    explicit ConfigValue(double value) : value_(value) {}

    double value() const { return value_; }

private:
    double value_ = 0.0;
};

class AnalysisReport {
public:
    AnalysisReport() = default;

    AnalysisReport& analyze() {
        analyzed_ = true;
        return *this;
    }

    AnalysisReport& report() {
        reported_ = true;
        return *this;
    }

    bool isAnalyzed() const { return analyzed_; }
    bool isReported() const { return reported_; }

private:
    bool analyzed_ = false;
    bool reported_ = false;
};

class TimePoint {
public:
    TimePoint() = default;

    double timestamp() const { return timestamp_; }

private:
    double timestamp_ = 0.0;
};

class TransformedElement {
public:
    TransformedElement() = default;

    TransformedElement& transform() {
        transformed_ = true;
        return *this;
    }

    bool isTransformed() const { return transformed_; }

private:
    bool transformed_ = false;
};

class ValidationResult {
public:
    ValidationResult() = default;

    ValidationResult& validate() {
        valid_ = true;
        return *this;
    }

    bool isValid() const { return valid_; }

private:
    bool valid_ = false;
};

// Multiple return member class
class MultiReturnProcessor {
public:
    MultiReturnProcessor() = default;

    // Multiple return types using std::tuple
    std::tuple<int, double, std::string> processValues(const std::vector<double>& input) {
        if (input.empty()) {
            return {0, 0.0, "empty"};
        }

        int count = static_cast<int>(input.size());
        double sum = std::accumulate(input.begin(), input.end(), 0.0);
        double average = sum / count;
        std::string status = average > 50.0 ? "high" : (average < 25.0 ? "low" : "normal");

        return {count, average, status};
    }

    // Multiple return types using struct
    struct ProcessResult {
        int success_count;
        double efficiency;
        std::string operation_type;
        bool was_successful;
    };

    ProcessResult processData(const std::vector<std::string>& operations) {
        ProcessResult result{};
        result.success_count = 0;
        result.efficiency = 0.0;
        result.operation_type = "unknown";
        result.was_successful = false;

        if (operations.empty()) {
            return result;
        }

        result.operation_type = "batch_processing";
        for (const auto& op : operations) {
            if (!op.empty()) {
                result.success_count++;
            }
        }

        result.efficiency = static_cast<double>(result.success_count) / operations.size();
        result.was_successful = result.efficiency > 0.7;

        return result;
    }

    // Multiple return values using output parameters
    void analyzeMetrics(const std::vector<int>& metrics, int& max_val, double& avg_val, std::string& category) {
        if (metrics.empty()) {
            max_val = 0;
            avg_val = 0.0;
            category = "empty";
            return;
        }

        max_val = *std::max_element(metrics.begin(), metrics.end());
        avg_val = std::accumulate(metrics.begin(), metrics.end(), 0.0) / metrics.size();

        if (avg_val > 75.0) {
            category = "excellent";
        } else if (avg_val > 50.0) {
            category = "good";
        } else if (avg_val > 25.0) {
            category = "fair";
        } else {
            category = "poor";
        }
    }
};

class Engine {
public:
    Engine()
        : pipeline_(), analyzer_(logger_, pipeline_), dashboard_{} {
        dashboard_.logView.logger = &logger_;
        // Initialize test data
        initializeTestData();
    }

    Pipeline& pipeline() { return pipeline_; }
    Pipeline* accessor() { return &pipeline_; }
    Analyzer& analyzer() { return analyzer_; }
    EngineDashboard& dashboard() { return dashboard_; }

    // Accessors for indexable test data
    DataArray<DataProcessor>& data() { return data_; }
    DataArray<Sensor>& sensors() { return sensors_; }
    DataMap<std::string, ConfigValue>& config() { return config_; }
    DataArray<AnalysisReport>& results() { return results_; }
    DataArray<DataArray<TimePoint>>& metrics() { return metrics_; }
    DataMap<std::string, DataProcessor>& maps() { return maps_; }
    DataArray<DataArray<TransformedElement>>& arrays() { return arrays_; }
    DataArray<DataMap<std::string, ValidationResult>>& collections() { return collections_; }

    // Accessor for multiple return processor
    MultiReturnProcessor& multiReturnProcessor() { return multiReturnProcessor_; }

    Engine& calibrate(std::function<double(double)> fn) {
        calibration_ = std::move(fn);
        return *this;
    }

    // Templated member function that can work with different numeric types
    template <typename T>
    T computeAggregate(const std::vector<T>& values, const std::string& operation) {
        if (values.empty()) {
            return T{};
        }

        if (operation == "sum") {
            return std::accumulate(values.begin(), values.end(), T{});
        } else if (operation == "average") {
            return std::accumulate(values.begin(), values.end(), T{}) / static_cast<T>(values.size());
        } else if (operation == "max") {
            return *std::max_element(values.begin(), values.end());
        } else if (operation == "min") {
            return *std::min_element(values.begin(), values.end());
        } else if (operation == "product") {
            return std::accumulate(values.begin(), values.end(), T{1}, std::multiplies<T>());
        } else {
            // Default to sum for unknown operations
            return std::accumulate(values.begin(), values.end(), T{});
        }
    }

    // Templated member function with two type parameters
    template <typename InputType, typename OutputType>
    std::vector<OutputType> transformValues(const std::vector<InputType>& input, std::function<OutputType(InputType)> transformer) {
        std::vector<OutputType> result;
        result.reserve(input.size());

        for (const auto& value : input) {
            result.push_back(transformer(value));
        }

        return result;
    }

    // Templated member function that filters based on a predicate
    template <typename T>
    std::vector<T> filterValues(const std::vector<T>& values, std::function<bool(T)> predicate) {
        std::vector<T> result;

        for (const auto& value : values) {
            if (predicate(value)) {
                result.push_back(value);
            }
        }

        return result;
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
    void initializeTestData() {
        // Initialize data arrays
        data_ = DataArray<DataProcessor>({DataProcessor{}, DataProcessor{}, DataProcessor{}});
        sensors_ = DataArray<Sensor>({Sensor{}, Sensor{}, Sensor{}, Sensor{}, Sensor{}, Sensor{}});

        // Initialize config map
        config_ = DataMap<std::string, ConfigValue>({
            {"threshold", ConfigValue{0.5}},
            {"limit", ConfigValue{100.0}},
            {"bias", ConfigValue{0.2}}
        });

        // Initialize results
        results_ = DataArray<AnalysisReport>({AnalysisReport{}, AnalysisReport{}});

        // Initialize metrics (nested arrays)
        metrics_ = DataArray<DataArray<TimePoint>>({
            DataArray<TimePoint>({TimePoint{}, TimePoint{}}),
            DataArray<TimePoint>({TimePoint{}, TimePoint{}, TimePoint{}}),
            DataArray<TimePoint>({TimePoint{}})
        });

        // Initialize maps
        maps_ = DataMap<std::string, DataProcessor>({
            {"key", DataProcessor{}},
            {"value", DataProcessor{}}
        });

        // Initialize arrays (nested arrays)
        arrays_ = DataArray<DataArray<TransformedElement>>({
            DataArray<TransformedElement>({TransformedElement{}, TransformedElement{}, TransformedElement{}, TransformedElement{}}),
            DataArray<TransformedElement>({TransformedElement{}, TransformedElement{}})
        });

        // Initialize collections (array of maps)
        collections_ = DataArray<DataMap<std::string, ValidationResult>>({
            DataMap<std::string, ValidationResult>({
                {"name", ValidationResult{}},
                {"type", ValidationResult{}}
            }),
            DataMap<std::string, ValidationResult>({
                {"id", ValidationResult{}},
                {"status", ValidationResult{}}
            })
        });
    }

    Logger logger_;
    Pipeline pipeline_;
    Analyzer analyzer_;
    EngineDashboard dashboard_;
    std::function<double(double)> calibration_;

    // Test data members for indexing
    DataArray<DataProcessor> data_;
    DataArray<Sensor> sensors_;
    DataMap<std::string, ConfigValue> config_;
    DataArray<AnalysisReport> results_;
    DataArray<DataArray<TimePoint>> metrics_;
    DataMap<std::string, DataProcessor> maps_;
    DataArray<DataArray<TransformedElement>> arrays_;
    DataArray<DataMap<std::string, ValidationResult>> collections_;

    // Multiple return processor
    MultiReturnProcessor multiReturnProcessor_;
};

}  // namespace callmaze

