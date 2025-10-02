#include <cmath>
#include <iostream>
#include <vector>

#include "call_maze.hpp"

int main() {
    using namespace callmaze;

    Engine engine;

    engine.pipeline().stage("primary").tune(1.4).limit(60.0);
    engine.pipeline().stage("secondary").tune(0.65).limit(40.0);

    auto descriptor = engine.accessor()->compound.describe();
    std::cout << descriptor << '\n';

    auto prepared = chain_value(
        tap(std::vector<double>{1.0, 4.0, 9.0, 16.0}, [](auto& vec) {
            vec.push_back(25.0);
            std::rotate(vec.begin(), vec.begin() + 1, vec.end());
        }),
        [](std::vector<double> vec) {
            std::transform(vec.begin(), vec.end(), vec.begin(), [](double value) {
                return std::sqrt(value);
            });
            return vec;
        },
        [](std::vector<double> vec) {
            vec.erase(vec.begin());
            return vec;
        });

    engine.analyzer().enable().setThreshold(0.5);
    engine.calibrate([](double value) { return value * 1.1; });

    auto value = engine.process(prepared);

    auto report = engine.accessor()->compound.sample(value);
    std::cout << report.summary() << '\n';

    auto tuningScore = engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize();
    std::cout << "Secondary tuning score: " << tuningScore << '\n';

    auto log = engine.dashboard().logView.print();
    std::cout << log;

    auto summary = engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary();
    std::cout << summary << '\n';

    return 0;
}

