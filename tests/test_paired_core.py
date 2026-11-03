"""Unit tests for scripts/paired_core.py (re-projection, matching, cluster statistics).

Run: python -m unittest discover -s tests -v      (from the repository root; CPU only)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
import paired_core as P  # noqa: E402


class SnapSym(unittest.TestCase):
    def test_extends_cut_surrogate(self):
        # original "10/03/2031" at a word boundary; difflib cut the surrogate to "0/03/2032"
        o, t = "on 10/03/2031 at", "on 10/03/2032 at"
        self.assertEqual(P.snap_sym(o, 3, 13, t, 4, 13), (3, 13))

    def test_recovers_longer_surrogate(self):
        # "Colombia)" became "Colombian)"; difflib kept only the shared prefix
        o, t = "(Colombia)", "(Colombian)"
        s, e = P.snap_sym(o, 1, 9, t, 1, 9)
        self.assertEqual(t[s:e], "Colombian")

    def test_respects_annotation_inside_word(self):
        # gold ends inside a word on the original: do not extend that side
        o, t = "Colombian food", "Colombian food"
        self.assertEqual(P.snap_sym(o, 0, 8, t, 0, 8), (0, 8))

    def test_no_change_at_boundaries(self):
        o, t = "Dr. Anna S. said", "Dr. Maria S. said"
        self.assertEqual(P.snap_sym(o, 4, 11, t, 4, 12), (4, 12))


class Matching(unittest.TestCase):
    g = (10, 20, "NOMBRE_SUJETO_ASISTENCIA")

    def m(self, ps, pe, label, crit):
        return P.matches(*self.g, {"start": ps, "end": pe, "label": label}, crit)

    def test_overlap(self):
        self.assertTrue(self.m(19, 25, "X", "overlap"))
        self.assertFalse(self.m(20, 25, "X", "overlap"))  # touching is not overlapping
        self.assertFalse(self.m(0, 10, "X", "overlap"))

    def test_iou50(self):
        self.assertTrue(self.m(10, 20, "X", "iou50"))
        self.assertTrue(self.m(12, 22, "X", "iou50"))     # 8 / 12 = 0.67
        self.assertFalse(self.m(15, 30, "X", "iou50"))    # 5 / 20 = 0.25

    def test_exact_and_type(self):
        self.assertTrue(self.m(10, 20, "DATE", "exact"))
        self.assertFalse(self.m(10, 21, "PERSON", "exact"))
        self.assertTrue(self.m(10, 20, "PERSON", "exact_type"))
        self.assertTrue(self.m(10, 20, "patient", "exact_type"))  # case-insensitive
        self.assertFalse(self.m(10, 20, "DATE", "exact_type"))
        self.assertFalse(self.m(10, 20, "SOMETHING_UNMAPPED", "exact_type"))

    def test_type_maps(self):
        self.assertEqual(P.gold_type("FECHAS"), "DATE")
        self.assertEqual(P.gold_type("TITLE"), "OTHER")
        self.assertEqual(P.pred_type("DATE_TIME"), "DATE")
        self.assertIsNone(P.pred_type("MEDICATION"))
        self.assertIsNone(P.pred_type(None))
        # every coarse type used by the maps is declared
        self.assertTrue(set(P.GOLD_TYPE.values()) | set(P.PRED_TYPE.values()) <= set(P.TYPES))


class ClusterStats(unittest.TestCase):
    def test_point_estimate_and_counts(self):
        per_doc = {"A": (np.array([2, 2]), np.array([2, 1]), np.array([1, 1])),
                   "B": (np.array([4]), np.array([4]), np.array([4]))}
        s = P.cluster_stats(per_doc, B=200)
        self.assertEqual(s["n"], 8)
        self.assertEqual(s["docs"], 3)
        self.assertAlmostEqual(s["rec_o"], 87.5)
        self.assertAlmostEqual(s["rec_t"], 75.0)
        self.assertAlmostEqual(s["diff"], 12.5)
        self.assertLessEqual(s["ci90"][0], s["diff"])
        self.assertGreaterEqual(s["ci90"][1], s["diff"])

    def test_zero_variance_is_equivalent(self):
        per_doc = {"A": (np.array([5] * 50), np.array([4] * 50), np.array([4] * 50))}
        s = P.cluster_stats(per_doc, B=200)
        self.assertEqual(s["diff"], 0.0)
        self.assertEqual(s["ci90"], [0.0, 0.0])
        self.assertTrue(s["tost"][1])
        self.assertEqual(s["p_rope"][1], 1.0)

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(1)
        n = rng.integers(1, 10, 40); fo = rng.binomial(n, .8); ft = rng.binomial(n, .78)
        a = P.cluster_stats({"A": (n, fo, ft)}, B=300, seed=7)
        b = P.cluster_stats({"A": (n, fo, ft)}, B=300, seed=7)
        self.assertEqual(a["ci90"], b["ci90"])

    def test_per_doc_arrays_sums(self):
        rows = [{"bench": "A", "doc": "d1", "o": {"overlap": True}, "t": {"overlap": False}},
                {"bench": "A", "doc": "d1", "o": {"overlap": True}, "t": {"overlap": True}},
                {"bench": "A", "doc": "d2", "o": {"overlap": False}, "t": {"overlap": True}}]
        n, fo, ft = P.per_doc_arrays(rows, "overlap")["A"]
        self.assertEqual(sorted(zip(n.tolist(), fo.tolist(), ft.tolist())), [(1, 0, 1), (2, 2, 1)])


@unittest.skipUnless(os.environ.get("RUN_DATA_TESTS") == "1", "set RUN_DATA_TESTS=1 to check against results/*.jsonl")
class PaperCounts(unittest.TestCase):
    """Denominators of the paper's headline (difflib re-projection, overlap)."""

    def test_headline_counts(self):
        rows = list(P.paired_rows("difflib"))
        self.assertEqual(len({(r["bench"], r["doc"], r["idx"]) for r in rows}), 5279)
        self.assertEqual(len({(r["bench"], r["doc"]) for r in rows}), 1147)
        self.assertEqual(len(rows), 57112)
        s = P.cluster_stats(P.per_doc_arrays(rows, "overlap"), B=500)
        self.assertEqual(round(s["rec_o"], 1), 76.1)
        self.assertEqual(round(s["rec_t"], 1), 74.9)


if __name__ == "__main__":
    unittest.main()
