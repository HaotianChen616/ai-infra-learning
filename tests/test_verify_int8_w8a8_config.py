import unittest

from labs.verify_int8_w8a8_config import verify_config


def valid_config() -> dict:
    return {
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
            "ignore": ["lm_head"],
        },
    }


class VerifyInt8W8A8ConfigTest(unittest.TestCase):
    def test_accepts_qwen3_compressed_tensors_int8_w8a8(self) -> None:
        result = verify_config(valid_config())
        self.assertTrue(result["valid"])
        self.assertEqual(result["format"], "int-quantized")
        self.assertEqual(len(result["matching_w8a8_groups"]), 1)

    def test_rejects_fp8_w8a8(self) -> None:
        config = valid_config()
        group = config["quantization_config"]["config_groups"]["group_0"]
        group["weights"]["type"] = "float"
        group["input_activations"]["type"] = "float"
        config["quantization_config"]["format"] = "float-quantized"
        result = verify_config(config)
        self.assertFalse(result["valid"])
        self.assertTrue(any("int-quantized" in item for item in result["failures"]))

    def test_rejects_weight_only_int8(self) -> None:
        config = valid_config()
        group = config["quantization_config"]["config_groups"]["group_0"]
        group["input_activations"] = None
        result = verify_config(config)
        self.assertFalse(result["valid"])
        self.assertTrue(any("input activations" in item for item in result["failures"]))


if __name__ == "__main__":
    unittest.main()
