#pragma once

#include "util.hpp"

inline flt sign(flt data) {
	if (data > 0.0f) {
		return 1.0f;
	}else {
		return -1.0f;
	}
}

template<typename T>
std::vector<T> sign(std::vector<T>& data) {
	auto len = data.size();
	auto ret = std::vector<T>(len);
	for (std::size_t i = 0; i < len; i++) {
		ret[i] = sign(data[i]);
	}
	return ret;
}

inline flt hard_tan(flt data) {
	return std::max(-1.0f, std::min(1.0f, data));
}

template<typename T>
std::vector<T> hard_tanh(std::vector<T>& data) {
	auto len = data.size();
	auto ret = std::vector<T>(len);
	for (std::size_t i = 0; i < len; i++) {
		ret[i] = hard_tanh(data[i]);
	}
	return ret;
}

inline flt hard_tanh_back(flt data) {
	if (-1.0f < data and data < 1.0f) {
		return 1.0f;
	}else {
		return 0.0f;
	}
}

template<typename T>
std::vector<T> hard_tanh_back(std::vector<T>& data) {
	auto len = data.size();
	auto ret = std::vector<T>(len);
	for (std::size_t i = 0; i < len; i++) {
		ret[i] = hard_tanh_back(data[i]);
	}
	return ret;
}

inline bool binarize(flt data) {
	return data > 0.0f;
}
