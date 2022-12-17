#include <bit>
#include <optional>

#include <openfhe.h>

#include "ndarray.h"

typedef lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Ciphertext;
typedef lbcrypto::CryptoContext<lbcrypto::DCRTPoly> CryptoContext;

static uint32_t ceil_log2(uint32_t value) {
	return 32 - std::countl_zero(value);
}

static Ciphertext inner_product(const CryptoContext &cc,
		const Ciphertext &a, const Ciphertext &b) {
	auto mult = cc->EvalMult(a, b);
	return cc->EvalSum(mult, cc->GetEncodingParams()->GetBatchSize());
}

static Ciphertext predict(const CryptoContext &cc, const Ciphertext &features,
		const Ciphertext &weights, const Ciphertext &bias,
		std::optional<Ciphertext*> intermediate = std::nullopt) {
	auto dot = inner_product(cc, features, weights);
	if (intermediate.has_value())
		**intermediate = dot;

	/* apply sigmoid using a least-squares approximation,
	 * degree-5 is picked to preserve the monotonicity of sigmoid
	 * coefficients from: https://doi.org/10.1186/s12920-018-0401-7 */
	const std::vector<double> coeffs = {
		0.5,							/* x^0 */
		1.53048 / 8,					/* x^1 */
		0,								/* x^2 */
		-2.3533056 / (8 * 8 * 8),		/* x^3 */
		0,								/* x^4 */
		1.3511295 / (8 * 8 * 8 * 8),	/* x^5 */
	};
	return cc->EvalPoly(cc->EvalAdd(dot, bias), coeffs);
}

static void print_usage(char *program) {
	fprintf(stderr, "usage: %s <image_idx>\n", program);
}

int main(int argc, char **argv) {
	if (argc != 1 && argc != 2) {
		print_usage(argv[0]);
		return -1;
	}

	const auto weights = ndarray<double>::load_from_file("lr_weights.bin",
			{ 28 * 28 + 1 });
	const auto images = ndarray<uint8_t>::load_from_idx_file(".data/mnist_trimmed/"
			"t10k-images-idx3-ubyte").to<double>();

	/* setup crypto context */
	const uint32_t mult_depth = 4;
	const uint32_t scale_mod_size = 50;
	const uint32_t batch_size = 1 << ceil_log2(28 * 28);
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(mult_depth);
	parameters.SetScalingModSize(scale_mod_size);
	parameters.SetBatchSize(batch_size);

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc =
		lbcrypto::GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension "
		<< cc->GetRingDimension() << std::endl;

	/* key generation */
	const auto keys = cc->KeyGen();
	std::cout << "generating multiplication keys" << std::endl;
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalSumKeyGen(keys.secretKey);

	std::cout << "encoding weights" << std::endl;
	const auto weight_span = weights[std::nullopt];
	const lbcrypto::Plaintext ptw = cc->MakeCKKSPackedPlaintext(
			weight_span.begin(), weight_span.end() - 1);
	const lbcrypto::Plaintext ptb = cc->MakeCKKSPackedPlaintext(
			weight_span.end() - 1, weight_span.end());

	std::cout << "encrypting weights" << std::endl;
	const Ciphertext ctw = cc->Encrypt(keys.publicKey, ptw);
	const Ciphertext ctb = cc->Encrypt(keys.publicKey, ptb);

	if (argc == 2) {
		std::size_t image_idx = std::strtoull(argv[1], NULL, 10);
		const auto image = images[image_idx];
		std::cout << "encoding image" << std::endl;
		const lbcrypto::Plaintext pti = cc->MakeCKKSPackedPlaintext(
				image.begin(), image.end());
		std::cout << "encrypting image" << std::endl;
		const Ciphertext cti = cc->Encrypt(keys.publicKey, pti);

		std::cout << "predicting..." << std::endl;
		Ciphertext before_sigmoid;
		const Ciphertext prediction = predict(cc, cti, ctw, ctb,
				&before_sigmoid);

		lbcrypto::Plaintext pt_before_sigmoid;
		cc->Decrypt(before_sigmoid, keys.secretKey, &pt_before_sigmoid);
		std::cout << "before sigmoid "
			<< pt_before_sigmoid->GetRealPackedValue()[0] << std::endl;

		lbcrypto::Plaintext pt_prediction;
		cc->Decrypt(prediction, keys.secretKey, &pt_prediction);
		std::cout << "prediction "
			<< pt_prediction->GetRealPackedValue()[0] << std::endl;

	} else {
		const auto labels = ndarray<uint8_t>::load_from_idx_file(
				".data/mnist_trimmed/t10k-labels-idx1-ubyte");
		std::size_t correct = 0;

		/* TODO: implement batching */
		for (std::size_t i = 0; i < images.shape[0]; i++) {
			const auto image = images[i];
			const uint8_t ground_truth = *labels[i].begin();

			std::cout << "predicting image " << i << std::endl;
			const lbcrypto::Plaintext pti = cc->MakeCKKSPackedPlaintext(
					image.begin(), image.end());
			const Ciphertext cti = cc->Encrypt(keys.publicKey, pti);

			const Ciphertext prediction = predict(cc, cti, ctw, ctb);

			lbcrypto::Plaintext pt_prediction;
			cc->Decrypt(prediction, keys.secretKey, &pt_prediction);
			if (((pt_prediction->GetRealPackedValue()[0] >= 0.5) ? 3 : 8) == ground_truth)
				correct++;
		}

		std::cout << "accuracy: "
			<< static_cast<double>(correct) / images.shape[0] << std::endl;
	}

	return 0;
}
