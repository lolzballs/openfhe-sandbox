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

static Ciphertext predict(const CryptoContext &cc,
		const Ciphertext &&features, const Ciphertext &&weights,
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
	return cc->EvalPoly(dot, coeffs);
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s <image_idx>\n", argv[0]);
		return -1;
	}
	std::size_t image_idx = std::strtoull(argv[1], NULL, 10);

	auto weights = ndarray<double>::load_from_file("lr_weights.bin", { 28 * 28 });
	auto images = ndarray<uint8_t>::load_from_idx_file(".data/mnist_trimmed/"
			"t10k-images-idx3-ubyte").to<double>();

	/* setup crypto context */
	uint32_t mult_depth = 4;
	uint32_t scale_mod_size = 50;
	uint32_t batch_size = 1 << ceil_log2(28 * 28);
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
	auto keys = cc->KeyGen();
	std::cout << "generating multiplication keys" << std::endl;
	cc->EvalMultKeysGen(keys.secretKey);

	cc->EvalSumKeyGen(keys.secretKey);

	std::cout << "encoding weights" << std::endl;
	auto weight_span = weights[std::nullopt];
	lbcrypto::Plaintext ptw = cc->MakeCKKSPackedPlaintext(
			weight_span.begin(), weight_span.end());

	auto image = images[image_idx];
	std::cout << "encoding image" << std::endl;
	lbcrypto::Plaintext pti = cc->MakeCKKSPackedPlaintext(
			image.begin(), image.end());

	std::cout << "encrypting weights" << std::endl;
	Ciphertext ctw = cc->Encrypt(keys.publicKey, ptw);
	std::cout << "encrypting image" << std::endl;
	Ciphertext cti = cc->Encrypt(keys.publicKey, pti);

	std::cout << "predicting..." << std::endl;
	Ciphertext before_sigmoid;
	Ciphertext prediction = predict(cc, std::move(cti), std::move(ctw),
			&before_sigmoid);

	lbcrypto::Plaintext pt_before_sigmoid;
	cc->Decrypt(before_sigmoid, keys.secretKey, &pt_before_sigmoid);
	std::cout << "intermediate " << " "
		<< pt_before_sigmoid->GetRealPackedValue()[0] << std::endl;

	lbcrypto::Plaintext pt_prediction;
	auto res = cc->Decrypt(prediction, keys.secretKey, &pt_prediction);
	std::cout << "prediction " << res.isValid
		<< " " << pt_prediction->GetRealPackedValue()[0] << std::endl;
}
