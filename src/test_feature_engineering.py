#!/usr/bin/env python3
"""
Amazon ML Hackathon: Business Entity Resolution Challenge
Stage 5 — Unit Tests for Feature Engineering Module

Verifies:
1. Feature list consistency (dict keys == FEATURE_NAMES)
2. Feature vector length and order
3. Exact name and address matches
4. Word order invariance (token sort and sorted token equality)
5. Minor typo robustness (RapidFuzz ratio and Jaro-Winkler)
6. Corporate legal suffix agreement and mismatch detection
7. Missing and empty address safety (no false signals on empty address)
8. Numeric address token matching (critical for Indic addresses)
9. Name and address evidence agreement and conflict indicators
10. Candidate generation provenance and priority score extraction
11. Multilingual handling (French accents, Devanagari Hindi)
12. Strict isolation: No entity ID leakage in features
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_preprocessing import preprocess_record
from feature_engineering import (
    FEATURE_NAMES,
    extract_pairwise_features,
    extract_feature_vector,
)


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Base fixture
        self.s1_base = preprocess_record({
            "entity_id": "S1-100",
            "business_name": "Apex Technology Solutions Inc",
            "business_address": "123 Main Street, Suite 400, Seattle, WA",
            "country": "US",
        })

    def test_feature_names_consistency(self):
        """Verifies that all extracted feature dict keys match FEATURE_NAMES exactly."""
        s2_match = preprocess_record({
            "entity_id": "S2-200",
            "business_name": "Apex Technology Solutions Incorporated",
            "business_address": "123 Main St, Ste 400, Seattle, WA",
            "country": "US",
        })
        feats = extract_pairwise_features(self.s1_base, s2_match)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertEqual(sorted(list(feats.keys())), sorted(FEATURE_NAMES))

        # Check vector extraction
        vec = extract_feature_vector(self.s1_base, s2_match)
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_exact_matches(self):
        """Verifies exact matches on normalized strings produce 1.0."""
        s2_exact = preprocess_record({
            "entity_id": "S2-201",
            "business_name": "Apex Technology Solutions Inc",
            "business_address": "123 Main Street, Suite 400, Seattle, WA",
            "country": "US",
        })
        feats = extract_pairwise_features(self.s1_base, s2_exact)
        self.assertEqual(feats["name_exact_norm"], 1.0)
        self.assertEqual(feats["name_exact_cons"], 1.0)
        self.assertAlmostEqual(feats["name_fuzz_ratio"], 1.0)
        self.assertEqual(feats["addr_exact_norm"], 1.0)
        self.assertEqual(feats["country_match"], 1.0)

    def test_word_order_variation(self):
        """Verifies token sort and sorted token equality handle word order permutations."""
        s1 = preprocess_record({
            "entity_id": "S1-102",
            "business_name": "Kumar Rajesh Enterprises",
            "business_address": "Plot 45, MG Road, Bengaluru",
            "country": "India",
        })
        s2 = preprocess_record({
            "entity_id": "S2-202",
            "business_name": "Rajesh Kumar Enterprises",
            "business_address": "Plot 45, MG Road, Bengaluru",
            "country": "India",
        })
        feats = extract_pairwise_features(s1, s2)
        self.assertEqual(feats["name_sorted_token_equal"], 1.0)
        self.assertAlmostEqual(feats["name_token_sort_ratio"], 1.0)
        self.assertAlmostEqual(feats["name_token_set_ratio"], 1.0)

    def test_spelling_variation(self):
        """Verifies character similarity catches typos."""
        s2_typo = preprocess_record({
            "entity_id": "S2-203",
            "business_name": "Apex Technlogy Solutns Inc",
            "business_address": "123 Main Street, Suite 400, Seattle, WA",
            "country": "US",
        })
        feats = extract_pairwise_features(self.s1_base, s2_typo)
        self.assertEqual(feats["name_exact_norm"], 0.0)
        self.assertGreater(feats["name_fuzz_ratio"], 0.85)
        self.assertGreater(feats["name_jaro_winkler"], 0.90)

    def test_corporate_suffix_features(self):
        """Verifies agreement and mismatch detection for legal suffixes."""
        # Exact suffix match (Inc vs Incorporated standardizes to inc)
        s2_match_sfx = preprocess_record({
            "entity_id": "S2-204",
            "business_name": "Apex Technology Solutions Corporation",
            "business_address": "123 Main St",
            "country": "US",
        })
        feats = extract_pairwise_features(self.s1_base, s2_match_sfx)
        self.assertEqual(feats["suffix_both_present"], 1.0)
        # S1 has 'inc', S2 has 'corp' -> mismatch!
        self.assertEqual(feats["suffix_mismatch"], 1.0)
        self.assertEqual(feats["suffix_exact_match"], 0.0)

        # Matching suffix
        s2_same_sfx = preprocess_record({
            "entity_id": "S2-205",
            "business_name": "Apex Technology Solutions Incorporated",
            "business_address": "123 Main St",
            "country": "US",
        })
        feats2 = extract_pairwise_features(self.s1_base, s2_same_sfx)
        self.assertEqual(feats2["suffix_exact_match"], 1.0)
        self.assertEqual(feats2["suffix_mismatch"], 0.0)

    def test_missing_and_empty_address_safety(self):
        """Verifies empty address safely sets indicators and zero string similarities."""
        s1_no_addr = preprocess_record({
            "entity_id": "S1-103",
            "business_name": "Global Logistics",
            "business_address": "",
            "country": "US",
        })
        s2_no_addr = preprocess_record({
            "entity_id": "S2-206",
            "business_name": "Global Logistics",
            "business_address": None,
            "country": "US",
        })
        feats = extract_pairwise_features(s1_no_addr, s2_no_addr)
        self.assertEqual(feats["addr_s1_empty"], 1.0)
        self.assertEqual(feats["addr_cand_empty"], 1.0)
        self.assertEqual(feats["addr_both_empty"], 1.0)
        self.assertEqual(feats["addr_either_empty"], 1.0)
        # String similarities must be strictly 0.0
        self.assertEqual(feats["addr_exact_norm"], 0.0)
        self.assertEqual(feats["addr_fuzz_ratio"], 0.0)
        self.assertEqual(feats["addr_token_jaccard"], 0.0)

    def test_numeric_address_token_match(self):
        """Verifies numeric house/building numbers are extracted and matched."""
        s1 = preprocess_record({
            "entity_id": "S1-104",
            "business_name": "Sri Krishna Sweets",
            "business_address": "Shop 314, 2nd Floor, Brigade Road, Bengaluru",
            "country": "India",
        })
        s2 = preprocess_record({
            "entity_id": "S2-207",
            "business_name": "Sri Krishna Sweets",
            "business_address": "314 Brigade Road",
            "country": "India",
        })
        feats = extract_pairwise_features(s1, s2)
        self.assertGreaterEqual(feats["addr_numeric_shared_count"], 1.0)
        self.assertGreaterEqual(feats["addr_numeric_jaccard"], 0.5)

    def test_evidence_agreement_and_conflict(self):
        """Verifies high agreement and conflict indicators."""
        # Strong name match, conflicting address
        s2_conflict = preprocess_record({
            "entity_id": "S2-208",
            "business_name": "Apex Technology Solutions Inc",
            "business_address": "999 Ocean Drive, Miami, Florida",
            "country": "US",
        })
        feats = extract_pairwise_features(self.s1_base, s2_conflict)
        self.assertEqual(feats["name_match_addr_conflict"], 1.0)
        self.assertEqual(feats["name_addr_high_agreement"], 0.0)

    def test_provenance_and_metadata(self):
        """Verifies candidate generation metadata is properly converted into features."""
        s2 = preprocess_record({
            "entity_id": "S2-209",
            "business_name": "Apex Technology",
            "business_address": "123 Main St",
            "country": "US",
        })
        metadata = {
            "blocking_methods": ["exact_name", "name_token", "addr_token"],
            "score": 145,
        }
        feats = extract_pairwise_features(self.s1_base, s2, cand_metadata=metadata)
        self.assertEqual(feats["prov_exact_name"], 1.0)
        self.assertEqual(feats["prov_name_token"], 1.0)
        self.assertEqual(feats["prov_addr_token"], 1.0)
        self.assertEqual(feats["prov_sorted_name"], 0.0)
        self.assertEqual(feats["prov_num_passes"], 3.0)
        self.assertAlmostEqual(feats["prov_blocking_score"], 145 / 150.0, places=2)

    def test_devanagari_and_french_handling(self):
        """Verifies multilingual entities are supported without degradation."""
        # French
        s1_fr = preprocess_record({
            "entity_id": "S1-FR",
            "business_name": "Société Générale Électrique",
            "business_address": "15 Rue de la Paix, Paris",
            "country": "France",
        })
        s2_fr = preprocess_record({
            "entity_id": "S2-FR",
            "business_name": "Societe Generale Electrique",
            "business_address": "15 Rue de la Paix, Paris",
            "country": "France",
        })
        feats_fr = extract_pairwise_features(s1_fr, s2_fr)
        self.assertEqual(feats_fr["name_exact_norm"], 1.0)

        # Devanagari Hindi
        s1_hi = preprocess_record({
            "entity_id": "S1-HI",
            "business_name": "स्टेट बैंक ऑफ़ इंडिया",
            "business_address": "मुंबई, महाराष्ट्र",
            "country": "India",
        })
        s2_hi = preprocess_record({
            "entity_id": "S2-HI",
            "business_name": "स्टेट बैंक ऑफ़ इंडिया",
            "business_address": "मुंबई, महाराष्ट्र",
            "country": "India",
        })
        feats_hi = extract_pairwise_features(s1_hi, s2_hi)
        self.assertEqual(feats_hi["name_exact_norm"], 1.0)
        self.assertEqual(feats_hi["addr_exact_norm"], 1.0)

    def test_strict_isolation_no_entity_id_leakage(self):
        """Confirms entity_id is never present in FEATURE_NAMES."""
        for fn in FEATURE_NAMES:
            self.assertNotIn("entity_id", fn.lower())
            self.assertNotIn("id", fn.lower().split("_"))


if __name__ == "__main__":
    unittest.main()
