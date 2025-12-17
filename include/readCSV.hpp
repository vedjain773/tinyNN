#ifndef READCSV
#define READCSV

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::vector <float>> readFile(const std::string& path, int noOfSamples);
std::vector<float> readLine(const std::string& path, int lineNum);
std::vector<float> sliceVector(std::vector<float> vec, double& label);

#endif
