#include <cmath>
#include <iostream>
#include <vector>
#include <memory>

#include "call_maze.hpp"

// Inheritance test classes
class Base {
public:
    virtual ~Base() = default;
    virtual std::string getName() const { return "Base"; }
    virtual int getValue() const { return 0; }
};

class Derived : public Base {
public:
    std::string getName() const override { return "Derived"; }
    int getValue() const override { return 42; }
    double getDerivedValue() const { return 3.14; }
};

class AnotherDerived : public Base {
public:
    std::string getName() const override { return "AnotherDerived"; }
    int getValue() const override { return 100; }
    std::string getSpecialValue() const { return "special"; }
};

// Factory functions for dynamic vs static type testing
Base* createBase() { return new Base(); }
Base* createDerived() { return new Derived(); }
Base* createAnotherDerived() { return new AnotherDerived(); }
Derived* createDerivedDirectly() { return new Derived(); }

// Function that returns derived type but base pointer
Base* factoryFunction() {
    return new Derived();
}

// Function that demonstrates polymorphic behavior
void processObject(Base* obj) {
    std::cout << "Processing: " << obj->getName() << " with value " << obj->getValue() << std::endl;
}

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

    auto tuningScore = engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize(); // go to definition for `finalize` does not always work
    std::cout << "Secondary tuning score: " << tuningScore << '\n';

    auto log = engine.dashboard().logView.print();
    std::cout << log;

    auto summary = engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary();
    std::cout << summary << '\n';

    // Inheritance and dynamic vs static type testing
    std::cout << "\n=== Inheritance and Type Testing ===\n";

    // Static type Base, dynamic type Base
    Base* baseObj = createBase();
    std::cout << "baseObj static: Base, dynamic: " << baseObj->getName() << std::endl;

    // Static type Base*, dynamic type Derived* (polymorphism)
    Base* derivedObjAsBase = createDerived();
    std::cout << "derivedObjAsBase static: Base*, dynamic: " << derivedObjAsBase->getName() << std::endl;

    // Static type Derived*, dynamic type Derived*
    Derived* derivedObjDirect = createDerivedDirectly();
    std::cout << "derivedObjDirect static: Derived*, dynamic: " << derivedObjDirect->getName() << std::endl;

    // Factory function returning Derived* as Base*
    Base* factoryObj = factoryFunction();
    std::cout << "factoryObj static: Base*, dynamic: " << factoryObj->getName() << std::endl;

    // Another derived type
    Base* anotherObj = createAnotherDerived();
    std::cout << "anotherObj static: Base*, dynamic: " << anotherObj->getName() << std::endl;

    // Type casting scenarios
    Derived* castObj = dynamic_cast<Derived*>(derivedObjAsBase);
    if (castObj) {
        std::cout << "castObj successful - can access derived value: " << castObj->getDerivedValue() << std::endl;
    }

    // Process objects polymorphically
    processObject(baseObj);
    processObject(derivedObjAsBase);
    processObject(anotherObj);

    // Cleanup
    delete baseObj;
    delete derivedObjAsBase;
    delete derivedObjDirect;
    delete factoryObj;
    delete anotherObj;

    // Indexing functionality test cases
    std::cout << "\n=== Indexing Functionality Testing ===\n";

    // Test 1: Simple array indexing with method call
    auto& dataProcessor = engine.data()[0];
    dataProcessor.process();
    std::cout << "DataProcessor[0] processed: " << dataProcessor.isProcessed() << std::endl;

    // Test 2: Numeric indexing with method call
    auto& sensor = engine.sensors()[5];
    sensor.calibrate();
    std::cout << "Sensors[5] calibrated: " << sensor.isCalibrated() << std::endl;

    // Test 3: The innermost function `get` returns a type paremeter
    auto& sensor = engine.sensors().get(5);
    sensor.calibrate();
    std::cout << "Sensors[5] calibrated: " << sensor.isCalibrated() << std::endl;

    // Test 4: String key indexing with method call
    auto& thresholdValue = engine.config()["threshold"];
    std::cout << "Config[\"threshold\"] value: " << thresholdValue.value() << std::endl;

    // Test 5: Variable indexing with chained calls
    int i = 0;
    auto& analysisReport = engine.results()[i];
    analysisReport.analyze().report();
    std::cout << "Results[i] analyzed and reported: " << analysisReport.isAnalyzed() << ", " << analysisReport.isReported() << std::endl;

    // Test 5: Nested indexing
    auto& timePoint = engine.metrics()[2][0];
    std::cout << "Metrics[2][0] timestamp: " << timePoint.timestamp() << std::endl;

    // Test 6: Map key indexing with chained calls
    auto& mapProcessor = engine.maps().get("key");
    mapProcessor.process();
    std::cout << "Maps[\"key\"] processed: " << mapProcessor.isProcessed() << std::endl;

    // Test 7: Multiple nested array indexing
    auto& element = engine.arrays()[1][3];
    element.transform();
    std::cout << "Arrays[1][3] transformed: " << element.isTransformed() << std::endl;

    // Test 8: Complex collection indexing (array of maps)
    auto& validation = engine.collections()[0]["name"];
    validation.validate();
    std::cout << "Collections[0][\"name\"] validated: " << validation.isValid() << std::endl;

    // Test Multiple Return Functions
    std::cout << "\n=== Multiple Return Functions Testing ===\n";

    // Test 1: Multiple return using std::tuple
    auto& multiProcessor = engine.multiReturnProcessor();
    std::vector<double> testValues = {10.5, 20.3, 30.7, 40.1, 50.9};
    auto tupleResult = multiProcessor.processValues(testValues);
    std::cout << "Tuple Result - Count: " << std::get<0>(tupleResult)
              << ", Average: " << std::get<1>(tupleResult)
              << ", Status: " << std::get<2>(tupleResult) << std::endl;

    // Test 2: Multiple return using struct
    std::vector<std::string> operations = {"process", "", "validate", "transform", "analyze"};
    auto structResult = multiProcessor.processData(operations);
    std::cout << "Struct Result - Success Count: " << structResult.success_count
              << ", Efficiency: " << structResult.efficiency
              << ", Operation: " << structResult.operation_type
              << ", Successful: " << (structResult.was_successful ? "Yes" : "No") << std::endl;

    // Test 3: Multiple return using output parameters
    std::vector<int> metrics = {85, 92, 78, 96, 88, 91, 83};
    int maxVal;
    double avgVal;
    std::string category;
    multiProcessor.analyzeMetrics(metrics, maxVal, avgVal, category);
    std::cout << "Output Parameters Result - Max: " << maxVal
              << ", Average: " << avgVal
              << ", Category: " << category << std::endl;

    // Test Templated Member Functions
    std::cout << "\n=== Templated Member Functions Testing ===\n";

    // Test 1: computeAggregate with different types
    std::vector<int> intData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> doubleData = {1.1, 2.2, 3.3, 4.4, 5.5};

    int intSum = engine.computeAggregate(intData, "sum");
    double doubleAvg = engine.computeAggregate(doubleData, "average");
    int intMax = engine.computeAggregate(intData, "max");
    double doubleMin = engine.computeAggregate(doubleData, "min");

    std::cout << "Template computeAggregate Results:\n";
    std::cout << "  int sum: " << intSum << std::endl;
    std::cout << "  double average: " << doubleAvg << std::endl;
    std::cout << "  int max: " << intMax << std::endl;
    std::cout << "  double min: " << doubleMin << std::endl;

    // Test 2: transformValues with different types
    auto squaredInts = engine.transformValues<int, int>(intData, [](int x) { return x * x; });
    auto doubledDoubles = engine.transformValues<double, double>(doubleData, [](double x) { return x * 2.0; });
    auto stringsFromInts = engine.transformValues<int, std::string>(intData, [](int x) {
        return "value_" + std::to_string(x);
    });

    std::cout << "Template transformValues Results:\n";
    std::cout << "  Squared ints: ";
    for (int val : squaredInts) std::cout << val << " ";
    std::cout << "\n  Doubled doubles: ";
    for (double val : doubledDoubles) std::cout << val << " ";
    std::cout << "\n  Strings from ints: ";
    for (const auto& str : stringsFromInts) std::cout << str << " ";
    std::cout << std::endl;

    // Test 3: filterValues with different types
    auto evenInts = engine.filterValues<int>(intData, [](int x) { return x % 2 == 0; });
    auto largeDoubles = engine.filterValues<double>(doubleData, [](double x) { return x > 3.0; });
    auto nonEmptyStrings = engine.filterValues<std::string>(operations, [](const std::string& s) { return !s.empty(); });

    std::cout << "Template filterValues Results:\n";
    std::cout << "  Even ints: ";
    for (int val : evenInts) std::cout << val << " ";
    std::cout << "\n  Large doubles (> 3.0): ";
    for (double val : largeDoubles) std::cout << val << " ";
    std::cout << "\n  Non-empty strings: ";
    for (const auto& str : nonEmptyStrings) std::cout << str << " ";
    std::cout << std::endl;

    // Test 4: Additional complex template function calls
    std::cout << "\n=== Additional Template Function Testing ===\n";

    // Test complex nested template calls
    auto processedInts = engine.transformValues<int, int>(intData, [](int x) { return x * 2; });
    auto filteredProcessed = engine.filterValues<int>(processedInts, [](int x) { return x > 10; });
    auto finalResult = engine.computeAggregate(filteredProcessed, "sum");

    std::cout << "Complex template pipeline result: " << finalResult << std::endl;

    // Test string manipulation templates
    std::vector<std::string> names = {"Alice", "Bob", "Charlie", "David"};
    auto nameLengths = engine.transformValues<std::string, int>(names, [](const std::string& name) {
        return name.length();
    });
    auto longNames = engine.filterValues<std::string>(names, [](const std::string& name) {
        return name.length() > 4;
    });
    auto totalLength = engine.computeAggregate(nameLengths, "sum");

    std::cout << "Name processing results:\n";
    std::cout << "  Total length: " << totalLength << std::endl;
    std::cout << "  Long names: ";
    for (const auto& name : longNames) std::cout << name << " ";
    std::cout << std::endl;

    // Test mixed type conversions
    auto intsToDoubles = engine.transformValues<int, double>(intData, [](int x) { return static_cast<double>(x); });
    auto doublesToStrings = engine.transformValues<double, std::string>(doubleData, [](double x) {
        return std::to_string(x);
    });

    std::cout << "Type conversion results:\n";
    std::cout << "  Ints to doubles count: " << intsToDoubles.size() << std::endl;
    std::cout << "  Doubles to strings count: " << doublesToStrings.size() << std::endl;

    return 0;
}

