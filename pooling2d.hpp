#pragma once

#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"

class AveragePooling2D : public Layer {
private:
	const std::size_t in_len;
	const std::size_t out_len;
	const std::size_t channels;
	const std::size_t in_height;
	const std::size_t in_width;
	const std::size_t padded_height;
	const std::size_t padded_width;
	const std::size_t out_height;
	const std::size_t out_width;
	const std::size_t kernel_height;
	const std::size_t kernel_width;
	const std::size_t kernel_area;
	const std::size_t h_padding;
	const std::size_t w_padding;
	const std::size_t h_stride;
	const std::size_t w_stride;
	flt& unpaddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
	flt& paddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
	flt& outputref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
public:
	AveragePooling2D(std::size_t ch,
					 std::size_t in_h, std::size_t in_w,
					 std::size_t out_h, std::size_t out_w,
					 std::size_t k_h, std::size_t k_w,
					 std::size_t s=1);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data); 
	inline void update(flt learningrate) {} // do nothing
};

class MaxPooling2D : public Layer {
private:
	const std::size_t in_len;
	const std::size_t out_len;
	const std::size_t channels;
	const std::size_t in_height;
	const std::size_t in_width;
	const std::size_t padded_height;
	const std::size_t padded_width;
	const std::size_t out_height;
	const std::size_t out_width;
	const std::size_t kernel_height;
	const std::size_t kernel_width;
	const std::size_t h_padding;
	const std::size_t w_padding;
	const std::size_t h_stride;
	const std::size_t w_stride;
	flt& unpaddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
	flt& paddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
	flt& outputref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
public:
	std::vector<std::vector<std::vector<std::vector<std::size_t>>>> h_selected;
	std::vector<std::vector<std::vector<std::vector<std::size_t>>>> w_selected;
	tensor_t paddedinputgrad;
	MaxPooling2D(std::size_t ch,
				 std::size_t in_h, std::size_t in_w,
				 std::size_t out_h, std::size_t out_w,
				 std::size_t k_h, std::size_t k_w,
				 std::size_t s=1);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	inline void update(flt learningrate) {} // do nothing
};
