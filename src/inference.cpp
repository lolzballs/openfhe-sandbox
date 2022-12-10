#include <openfhe.h>

#include "ndarray.h"

int main(int argc, char **argv) {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	auto weights = ndarray<double>::load_from_file("lr_weights.bin", { 28 * 28 });

	std::cout << weights[{ 600 }] << std::endl;
}
