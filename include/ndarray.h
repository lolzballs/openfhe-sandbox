#ifndef NDARRAY_H
#define NDARRAY_H

#include <cassert>
#include <endian.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>


template<typename T>
class ndarray {
public:
	std::vector<std::size_t> shape;

	ndarray(const std::vector<std::size_t> &shape)
		: shape(shape) {
		const std::size_t size = calculate_size(shape);
		backing.resize(size);
	};

	ndarray(const std::vector<std::size_t> &shape,
			const std::vector<uint8_t> &bytes)
		: shape(shape) {
		const std::size_t size = calculate_size(shape);
		assert(bytes.size() == size * sizeof(T));

		backing.resize(size);
		memcpy(&backing[0], &shape[0], bytes.size());
	};

	ndarray(const std::vector<std::size_t> &shape,
			const std::vector<T> &&contents)
		: shape(shape),
		backing(std::move(contents)) {
		const std::size_t size = calculate_size(shape);
		assert(contents.size() == size);
	};

	const T &operator[](const std::vector<std::size_t> &index) const {
		return backing[flatten_idx(index)];
	}

	T &operator[](const std::vector<std::size_t> &index) {
		return backing[flatten_idx(index)];
	}

	std::vector<T> operator[](std::optional<std::size_t> major_idx) const {
		/* TOOD: make this more efficient without copying */
		if (!major_idx.has_value())
			return backing;

		std::vector<std::size_t> minor_shape;
		minor_shape.reserve(shape.size() - 1);
		for (std::size_t i = 0; i < shape.size() - 1; i++) {
			minor_shape.push_back(shape[i + 1]);
		}
		std::size_t minor_size = calculate_size(minor_shape);

		auto start = backing.begin() + minor_size * *major_idx;
		return std::vector(start, start + minor_size);
	}

	static ndarray<T> load_from_istream(std::ifstream &is,
			const std::vector<std::size_t> &&shape) {
		const std::size_t size = calculate_size(shape);
		std::vector<T> data(size);
		const std::size_t size_in_bytes = size * sizeof(T);

		is.read(reinterpret_cast<char*>(data.data()), size_in_bytes);
		/* make sure we read the expected values */
		assert(static_cast<std::size_t>(is.gcount()) == size_in_bytes);

		return ndarray(shape, std::move(data));
	}

	static ndarray<T> load_from_file(std::string const& filename,
			const std::vector<std::size_t> &&shape) {
		std::ifstream is(filename, std::ios::binary);
		return load_from_istream(is, std::move(shape));
	}

	static ndarray<uint8_t> load_from_idx_file(std::string const& filename) {
		std::ifstream is(filename, std::ios::binary);

		uint16_t magic;
		is.read(reinterpret_cast<char*>(&magic), sizeof(uint16_t));
		uint8_t type;
		is.read(reinterpret_cast<char*>(&type), sizeof(uint8_t));
		uint8_t dim_count;
		is.read(reinterpret_cast<char*>(&dim_count), sizeof(uint8_t));

		assert(magic == 0x0000);
		assert(type == 0x08); /* uint8 */

		std::vector<std::size_t> shape(dim_count);
		is.read(reinterpret_cast<char*>(shape.data()),
				sizeof(uint32_t) * dim_count);
		for (auto &dim : shape) {
			dim = be32toh(dim);
		}

		return load_from_istream(is, std::move(shape));
	}

private:
	std::vector<T> backing;

	std::size_t flatten_idx(const std::vector<std::size_t> &index) const {
		assert(index.size() == shape.size());

		std::size_t column = 1;
		std::size_t idx = 0;
		for (std::size_t i = 0; i < index.size(); i++) {
			idx += index[i] * column;
			column *= shape[i];
		}
		std::cout << idx << std::endl;
		return idx;
	}

	static std::size_t calculate_size(const std::vector<std::size_t> &shape) {
		std::size_t size = 1;
		for (std::size_t dim_size : shape) {
			size *= dim_size;
		}
		return size;
	}
};

#endif
