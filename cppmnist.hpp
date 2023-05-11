#ifndef CPPMNIST_HPP
#define CPPMNIST_HPP

#include <bit>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <concepts>

namespace MNIST{

std::uint32_t swapendian(std::uint32_t x) {
	return x>>24 | (x<<8)&0x00ff0000 | (x>>8)&0x0000ff00 | x<<24;
}

template<typename T=std::uint8_t>
class Images{
private:
	std::vector<std::vector<std::vector<T>>> data_;
	const std::string filename_;
	std::uint32_t ftype_;
	std::uint32_t num_;
	std::uint32_t width_;
	std::uint32_t height_;
	const std::uint8_t threshold = 128u;
	std::ifstream ifs;
	void read_header();
	void read_payload();
public:
	const std::vector<std::vector<std::vector<T>>>& data() const;
	Images(std::string filename);
	~Images();
	void dump(const std::size_t idx, std::ostream& ost=std::cout) const;
};


template<typename T>
Images<T>::Images(std::string filename): filename_(filename) {
	this->ifs = std::ifstream(this->filename_, std::ios::in | std::ios::binary);
	if (!this->ifs.is_open()) {
		std::stringstream ss;
		ss << "Failed to read \"" << this->filename_
		   << "\"." << std::endl;
		throw std::runtime_error(ss.str());
	}
	this->read_header();
	this->read_payload();
	this->ifs.close();
}

template<typename T>
Images<T>::~Images() {
}

template<typename T>
void Images<T>::dump(const std::size_t idx, std::ostream& ost) const {
	for (std::size_t i = 0; i < this->height_; i++) {
		for (std::size_t j = 0; j < this->width_; j++) {
			if constexpr(std::is_integral_v<T>) {
				ost << (this->data_.at(idx)[i][j] < this->threshold ? "  " : "**");
			}else {
				ost << (this->data_.at(idx)[i][j] < ((T)this->threshold/255) ? "  " : "**");
			}
		}
		std::cout << std::endl;
	}
}

template<>
void Images<bool>::dump(const std::size_t idx, std::ostream& ost) const {
	for (std::size_t i = 0; i < this->height_; i++) {
		for (std::size_t j = 0; j < this->width_; j++) {
			ost << (this->data_.at(idx)[i][j] ? "**": "  ");
		}
		ost << std::endl;
	}
}

template<typename T>
const std::vector<std::vector<std::vector<T>>>& Images<T>::data() const{
	return this->data_;
}

template<typename T>
void Images<T>::read_header() {
	this->ifs.read((char*)&this->ftype_, sizeof(this->ftype_));
	this->ifs.read((char*)&this->num_, sizeof(this->num_));
	this->ifs.read((char*)&this->width_, sizeof(this->width_));
	this->ifs.read((char*)&this->height_, sizeof(this->height_));
	if constexpr(std::endian::native == std::endian::little) {
		this->ftype_ = swapendian(this->ftype_);
		this->num_ = swapendian(this->num_);
		this->width_ = swapendian(this->width_);
		this->height_ = swapendian(this->height_);
	}
	if (this->ftype_ != 0x803u) {
		std::stringstream ss;
		ss << "This file is not a MNIST images file." << std::endl;
		throw std::runtime_error(ss.str());
	}
}

template<typename T>
void Images<T>::read_payload() {
	this->data_.resize(this->num_);
	for (auto& image: this->data_) {
		image.resize(this->height_);
		for (auto& row: image) {
			row.resize(this->width_);
			for (auto& pixel: row) {
				std::uint8_t tmp;
				this->ifs.read((char*)&tmp, sizeof(tmp));
				if constexpr(std::is_integral_v<T>) {
					pixel = tmp;
				}else {
					pixel = (T)tmp/255;
				}
			}
		}
	}
}

template<>
void Images<bool>::read_payload() {
	this->data_.resize(this->num_);
	for (auto& image: this->data_) {
		image.resize(this->height_);
		for (auto& row: image) {
			row.resize(this->width_);
			for (std::size_t i = 0; i < row.size(); i++) {
				std::uint8_t tmp;
				this->ifs.read((char*)&tmp, sizeof(tmp));
				row[i] = tmp >= this->threshold;
			}
		}
	}
}

template<std::integral T=int, typename U=double>
class Labels{
private:
	std::vector<T> data_;
	const std::string filename_;
	std::uint32_t ftype_;
	std::uint32_t num_;
	void read();
public:
	const std::vector<T>& data() const;
	std::vector<std::vector<U>> onehot() const;
	Labels(std::string filename);
};

template<typename T, typename U>
Labels<T, U>::Labels(std::string filename): filename_(filename) {
	this->read();
}

template<typename T, typename U>
const std::vector<T>& Labels<T, U>::data() const {
	return this->data_;
}

template<typename T, typename U>
std::vector<std::vector<U>> Labels<T, U>::onehot() const {
	std::vector<std::vector<U>> mat(this->data_.size(), std::vector<U>(10, 0.0f));
	for (std::size_t i = 0; i < this->data_.size(); i++) {
		mat[i][this->data_[i]] = 1.0f;
	}
	return mat;
}

template<typename T, typename U>
void Labels<T, U>::read() {
	std::ifstream ifs(this->filename_, std::ios::in | std::ios::binary);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "Failed to read \"" << this->filename_
		   << "\"." << std::endl;
		throw std::runtime_error(ss.str());
	}
	ifs.read((char*)&this->ftype_, sizeof(this->ftype_));
	ifs.read((char*)&this->num_, sizeof(this->num_));
	if constexpr(std::endian::native == std::endian::little) {
		this->ftype_ = swapendian(this->ftype_);
		this->num_ = swapendian(this->num_);
	}
	if (this->ftype_ != 0x801u) {
		std::stringstream ss;
		ss << "This file is not a MNIST labels file." << std::endl;
		throw std::runtime_error(ss.str());
	}
	this->data_.resize(this->num_);
	for (auto& label: this->data_) {
		std::uint8_t tmp;
		ifs.read((char*)&tmp, sizeof(tmp));
		label = tmp;
	}
	ifs.close();
}

} // namespace mnist

#endif // include guard of CPPMNIST_HPP
