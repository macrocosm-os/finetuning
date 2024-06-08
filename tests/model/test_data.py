import unittest
from model.data import ModelId


class TestData(unittest.TestCase):
    def test_model_id_compressed_string(self):
        """Verifies a model_id can be compressed and decompressed."""
        # Note: The hash is excluded since it isn't included in the compressed string.
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            secure_hash="secure_hash",
            competition_id="12",
        )

        roundtrip_model_id = ModelId.from_compressed_str(model_id.to_compressed_str())

        self.assertEqual(model_id, roundtrip_model_id)

    def test_model_id_compressed_string_no_commit(self):
        """Verifies a model_id without a commit can be compressed and decompressed."""
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            secure_hash="secure_hash",
            competition_id="12",
        )

        roundtrip_model_id = ModelId.from_compressed_str(model_id.to_compressed_str())

        self.assertEqual(model_id, roundtrip_model_id)


if __name__ == "__main__":
    unittest.main()
