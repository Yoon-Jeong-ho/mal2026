import unittest

from scripts.merge_v3_tail_rationale_sft import normalized_target, remapped_key


class MergeV3TailRationaleSFTTests(unittest.TestCase):
    def test_remapped_keys_bind_handoff(self):
        self.assertNotEqual(remapped_key("replicate-001", "same"), remapped_key("replicate-002", "same"))
        self.assertEqual(remapped_key("replicate-001", "same"), remapped_key("replicate-001", "same"))

    def test_target_normalization_ignores_only_whitespace(self):
        first = {"content": {"rationale": "가 나"}}
        second = {"content": {"rationale": "가나"}}
        third = {"content": {"rationale": "가다"}}
        self.assertEqual(normalized_target(first), normalized_target(second))
        self.assertNotEqual(normalized_target(first), normalized_target(third))


if __name__ == "__main__":
    unittest.main()
