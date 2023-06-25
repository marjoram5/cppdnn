#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <algorithm>

using flt = float;
using vec_t = std::vector<flt>;
using tensor_t = std::vector<vec_t>;

template<typename T>
std::ostream& operator<<(std::ostream& ost, std::vector<T> vec) {
	ost << "[";
	for (std::size_t i = 0; i < vec.size(); i++) {
		ost << vec[i] << (i != vec.size()-1 ? ", ": "");
	}
	ost << "]";
	return ost;
}
