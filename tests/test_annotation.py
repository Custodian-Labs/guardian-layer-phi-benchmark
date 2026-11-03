"""Unit tests for the annotation aggregation in scripts/analyze_human_annotation.py (synthetic fixtures only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
import analyze_human_annotation as H  # noqa: E402


class Kappa(unittest.TestCase):
    def test_perfect(self):
        po, k = H.kappa(["a", "b", "a"], ["a", "b", "a"])
        self.assertEqual((po, k), (1.0, 1.0))

    def test_chance_level(self):
        po, k = H.kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
        self.assertAlmostEqual(po, 0.5); self.assertAlmostEqual(k, 0.0)

    def test_known_value(self):
        # 2x2 table [[20, 5], [10, 15]]: po = .70, pe = .50, kappa = .40
        a = ["y"] * 25 + ["n"] * 25
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        po, k = H.kappa(a, b)
        self.assertAlmostEqual(po, 0.70); self.assertAlmostEqual(k, 0.40)

    def test_normalises_case_and_space(self):
        self.assertEqual(H.kappa([" Well_formed"], ["well_formed "])[0], 1.0)


class Consensus(unittest.TestCase):
    def test_agree_disagree_adjudicate(self):
        a = {"1": {"category": "x_masked"}, "2": {"category": "well_formed"}, "3": {"category": "well_formed"}}
        b = {"1": {"category": "x_masked"}, "2": {"category": "salience_change"}, "3": {"category": "type_inconsistent"}}
        self.assertEqual(H.consensus(a, b), {"1": "x_masked", "2": "disagreement", "3": "disagreement"})
        adj = {"2": {"category": "salience_change"}}
        self.assertEqual(H.consensus(a, b, adj)["2"], "salience_change")

    def test_buckets(self):
        self.assertEqual(H.bucket("x_masked", False), "generator defect")
        self.assertEqual(H.bucket("well_formed", False), "detector sensitivity")
        self.assertEqual(H.bucket("well_formed", True), "span-mapping artefact")
        self.assertEqual(H.bucket("disagreement", False), "uncertain")


class BootRate(unittest.TestCase):
    def test_constant_rate(self):
        units = [(("b", f"d{i}"), 1, 1) if i % 2 else (("b", f"d{i}"), 0, 1) for i in range(40)]
        lo, hi = H.boot_rate(units, B=300)
        self.assertLessEqual(lo, 50); self.assertGreaterEqual(hi, 50)


if __name__ == "__main__":
    unittest.main()
