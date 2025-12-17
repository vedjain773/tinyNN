#include <readCSV.hpp>

std::vector<std::vector<float>> readFile(const std::string& filename, int noOfSamples) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file");
    }

    std::vector<std::vector<float>> data;
    data.reserve(noOfSamples);

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        row.reserve(785);

        const char* ptr = line.c_str();
        char* end;

        while (*ptr) {
            float value = std::strtof(ptr, &end);
            row.push_back(value);
            ptr = (*end == ',') ? end + 1 : end;
        }

        data.push_back(std::move(row));
    }

    return data;
}

std::vector<float> readLine(const std::string& path, int lineNum) {
    std::vector<float> row;
    std::ifstream file(path);
    std::string line;

    row.reserve(785);

    int counter = 0;
    while (std::getline(file, line)) {
        if (counter == lineNum) {
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stof(cell));
            }

            break;
        }
        counter++;
    }
    return row;
}

std::vector<float> sliceVector(std::vector<float> vec, double& label) {
    label = vec.at(0);

    std::vector<float> newVec;

    for (int i = 1; i < vec.size(); i++) {
        newVec.push_back(vec.at(i));
    }

    return newVec;
}
