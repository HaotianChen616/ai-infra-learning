import json
import tempfile
import unittest
from pathlib import Path

from labs.verify_ascend_w8a8_model import verify_model_directory


class VerifyAscendW8A8ModelTest(unittest.TestCase):
    def _model_dir(self, root: Path, *, quant_value: str) -> Path:
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "quant_model_description.json").write_text(
            json.dumps({"model.layers.0.q_proj.weight": quant_value}),
            encoding="utf-8",
        )
        (root / "quant_model_weights-00001-of-00001.safetensors").touch()
        (root / "quant_model_weights.safetensors.index.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return root

    def test_accepts_qwen36_modelslim_w8a8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_model_directory(
                self._model_dir(Path(directory), quant_value="W8A8_DYNAMIC")
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["w8a8_description_entries"], 1)

    def test_rejects_checkpoint_without_w8a8_description(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_model_directory(
                self._model_dir(Path(directory), quant_value="FLOAT")
            )
        self.assertFalse(result["valid"])
        self.assertTrue(any("no W8A8" in item for item in result["failures"]))


if __name__ == "__main__":
    unittest.main()
