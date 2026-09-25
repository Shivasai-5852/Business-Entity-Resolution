#!/usr/bin/env python3
"""
Amazon ML Hackathon: Business Entity Resolution Challenge
Stage 4 — Unit Tests for Candidate Blocking and Generation Module

Verifies all candidate generation behaviors using synthetic fixtures:
1. Exact normalized-name matches
2. Distinctive name token matches
3. Character n-gram prefix matches
4. Address-based candidate generation
5. Missing and empty addresses safety (never match on empty address)
6. Empty business names safety (never match on empty names)
7. Country partition isolation (US never matches India/France)
8. French accented names support
9. Devanagari (Hindi) names support
10. Deduplication & multi-pass provenance tracking
11. Deterministic output ordering
12. Candidate caps and oversized block handling
13. Ground-truth recall evaluation function accuracy
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_preprocessing import preprocess_record
from candidate_generation import (
    CandidateIndex,
    CandidateGenerator,
    evaluate_blocking_recall,
)


class TestCandidateGeneration(unittest.TestCase):
    """Synthetic unit tests for candidate blocking mechanisms."""

    def setUp(self):
        # Target dataset (e.g. Source 2)
        self.target_records = [
            {
                "entity_id": "S2-001",
                "business_name": "Zephay Labs Inc",
                "business_address": "2621 Cotten Road, Tyler, TX",
                "country": "US",
            },
            {
                "entity_id": "S2-002",
                "business_name": "Brahma Infosoft Pvt Ltd",
                "business_address": "Coimbatore Colony, Hunsur, Mysore, Karnataka",
                "country": "India",
            },
            {
                "entity_id": "S2-003",
                "business_name": "Marina Ecole France Sarl",
                "business_address": "63 R. DE DIEPPE, LILLE, Hauts-de-France",
                "country": "France",
            },
            {
                "entity_id": "S2-004",
                "business_name": "एसएस फूड",  # Hindi name
                "business_address": "AF 684 Nandgram Near Mother India School, Ghaziabad",
                "country": "India",
            },
            {
                "entity_id": "S2-005",
                "business_name": "मॉडर्न फाइनेंस प्राइवेट लिमिटेड",
                "business_address": "",  # Empty address
                "country": "India",
            },
            {
                "entity_id": "S2-006",
                "business_name": "Generic Retail Store",
                "business_address": "",  # Empty address
                "country": "US",
            },
            {
                "entity_id": "S2-007",
                "business_name": "Apple Inc",
                "business_address": "1 Infinite Loop, Cupertino, CA",
                "country": "US",
            },
            {
                "entity_id": "S2-008",
                "business_name": "Dahlia Ponr Reliable Scientific",  # Typo in 'Power'
                "business_address": "630 45th Ter, Kansas City, MO",
                "country": "US",
            },
        ]

        self.index = CandidateIndex(source_name="Source2").build_from_records(self.target_records)
        self.generator = CandidateGenerator(self.index, max_candidates_per_entity=10)

    def test_exact_normalized_name_match(self):
        # Query S1 has "Zephay Labs Corporation" -> normalized "zephay labs"
        # S2 has "Zephay Labs Inc" -> normalized "zephay labs"
        query = {
            "entity_id": "S1-101",
            "business_name": "Zephay Labs Corporation",
            "business_address": "Tyler, Texas",
            "country": "US",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-001", cand_ids)

        match = next(c for c in cands if c["candidate_entity_id"] == "S2-001")
        self.assertIn("exact_name", match["blocking_methods"])

    def test_distinctive_token_match(self):
        # Query S1 shares distinctive token 'infosoft' with S2-002
        query = {
            "entity_id": "S1-102",
            "business_name": "Brahma Infosoft Technologies",
            "business_address": "Bangalore",
            "country": "India",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-002", cand_ids)

        match = next(c for c in cands if c["candidate_entity_id"] == "S2-002")
        self.assertIn("name_token", match["blocking_methods"])

    def test_character_prefix_match(self):
        # S1 "Dahlia Power Reliable" matches S2-008 "Dahlia Ponr..." via prefix "dahliap"
        query = {
            "entity_id": "S1-103",
            "business_name": "Dahlia Power Solutions",
            "business_address": "Kansas City",
            "country": "US",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-008", cand_ids)

    def test_address_based_cross_lingual_match(self):
        # Query S1 has English name "SS Food", S2-004 has Hindi name "एसएस फूड"
        # Names do not match, but address shares 'nandgram' and 'ghaziabad'
        query = {
            "entity_id": "S1-104",
            "business_name": "SS Food Company",
            "business_address": "Plot 12, Nandgram, Ghaziabad, UP",
            "country": "India",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-004", cand_ids)

        match = next(c for c in cands if c["candidate_entity_id"] == "S2-004")
        self.assertIn("addr_token", match["blocking_methods"])

    def test_empty_address_safety(self):
        # Query S1 has an empty address. S2-005 and S2-006 also have empty addresses.
        # S1 MUST NOT match S2-006 purely because both have empty addresses!
        query = {
            "entity_id": "S1-105",
            "business_name": "Completely Different Unrelated Firm",
            "business_address": "",
            "country": "US",
        }
        cands = self.generator.generate_candidates(query)
        # Should be empty since names and tokens are completely different
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertNotIn("S2-006", cand_ids)
        self.assertEqual(len(cands), 0)

    def test_empty_business_name_safety(self):
        # Query S1 has empty name
        query = {
            "entity_id": "S1-106",
            "business_name": "",
            "business_address": "Cupertino, CA",
            "country": "US",
        }
        cands = self.generator.generate_candidates(query)
        # Empty name must not match S2-007 or any other
        for c in cands:
            self.assertNotIn("exact_name", c["blocking_methods"])

    def test_different_country_partitions(self):
        # S1 is in US with name "Brahma Infosoft".
        # Target S2-002 has name "Brahma Infosoft Pvt Ltd" in India.
        # Country partitioning must strictly prevent cross-border match!
        query = {
            "entity_id": "S1-107",
            "business_name": "Brahma Infosoft",
            "business_address": "Austin, Texas",
            "country": "US",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertNotIn("S2-002", cand_ids)

    def test_french_accented_names(self):
        # S1 has accented French name "Marina École France"
        # S2 has "Marina Ecole France Sarl"
        query = {
            "entity_id": "S1-108",
            "business_name": "Marina École France",
            "business_address": "Lille",
            "country": "France",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-003", cand_ids)

        match = next(c for c in cands if c["candidate_entity_id"] == "S2-003")
        self.assertIn("exact_name", match["blocking_methods"])

    def test_devanagari_names(self):
        # S1 has Devanagari name "मॉडर्न फाइनेंस" in India
        query = {
            "entity_id": "S1-109",
            "business_name": "मॉडर्न फाइनेंस",
            "business_address": "Gurugram, Haryana",
            "country": "India",
        }
        cands = self.generator.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-005", cand_ids)

    def test_duplicate_removal_and_provenance(self):
        # S1 matches S2-001 on BOTH exact_name AND address
        query = {
            "entity_id": "S1-110",
            "business_name": "Zephay Labs Inc",
            "business_address": "2621 Cotten Road, Tyler, TX",
            "country": "US",
        }
        cands = self.generator.generate_candidates(query)
        matching_s2_001 = [c for c in cands if c["candidate_entity_id"] == "S2-001"]
        # Exactly one candidate record for S2-001 (deduplicated)
        self.assertEqual(len(matching_s2_001), 1)
        methods = matching_s2_001[0]["blocking_methods"]
        # Provenance records multiple passes
        self.assertIn("exact_name", methods)
        self.assertIn("name_token", methods)
        self.assertIn("addr_token", methods)

    def test_deterministic_output(self):
        query = {
            "entity_id": "S1-111",
            "business_name": "Zephay Labs Inc",
            "business_address": "Cotten Road",
            "country": "US",
        }
        res1 = self.generator.generate_candidates(query)
        res2 = self.generator.generate_candidates(query)
        self.assertEqual(res1, res2)

    def test_candidate_caps(self):
        # Test generator with cap of 2
        strict_gen = CandidateGenerator(self.index, max_candidates_per_entity=2)
        query = {
            "entity_id": "S1-112",
            "business_name": "Retail Store Apple Inc",
            "business_address": "Cupertino Road",
            "country": "US",
        }
        cands = strict_gen.generate_candidates(query)
        self.assertLessEqual(len(cands), 2)

    def test_ground_truth_recall_calculation(self):
        queries = [
            {"entity_id": "S1-1", "business_name": "Zephay Labs", "business_address": "", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Brahma Infosoft", "business_address": "", "country": "India"},
            {"entity_id": "S1-3", "business_name": "Nonexistent Corp", "business_address": "", "country": "US"},
        ]
        # Ground truth: S1-1 matches S2-001, S1-2 matches S2-002, S1-3 has 0 matches
        gt_map = {
            "S1-1": {"S2-001"},
            "S1-2": {"S2-002"},
            "S1-3": set(),
        }
        stats = evaluate_blocking_recall(queries, self.generator, gt_map)
        self.assertEqual(stats["total_gt_matches"], 2)
        self.assertEqual(stats["retrieved_gt_matches"], 2)
        self.assertEqual(stats["candidate_recall_pct"], 100.0)
        self.assertEqual(stats["gt_lost_to_blocking"], 0)
        self.assertEqual(stats["gt_lost_to_cap"], 0)

    def test_numeric_address_token_match(self):
        # S1 has different name but shares numeric house token '729' and '11'
        query = {
            "entity_id": "S1-113",
            "business_name": "Completely Different Name",
            "business_address": "729 Cr 11, Tiffin, OH",
            "country": "US",
        }
        # Add a target with '729 Cr 11'
        target = {
            "entity_id": "S2-009",
            "business_name": "Target Legacy",
            "business_address": "729 CR 11, Clinton, OH",
            "country": "US",
        }
        idx = CandidateIndex(source_name="Source2").build_from_records(self.target_records + [target])
        gen = CandidateGenerator(idx, max_candidates_per_entity=10)
        cands = gen.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-009", cand_ids)
        match = next(c for c in cands if c["candidate_entity_id"] == "S2-009")
        self.assertIn("addr_token", match["blocking_methods"])

    def test_dynamic_in_flight_pruning(self):
        # If max_name_df=2, the 3rd occurrence must trigger pruning
        records = [
            {"entity_id": f"S2-T{i}", "business_name": f"Universal Venture {i}", "business_address": "", "country": "US"}
            for i in range(5)
        ]
        pruned_idx = CandidateIndex(source_name="Source2", max_name_df=2).build_from_records(records)
        # 'universal' appeared 5 times > max_name_df=2 -> should be in high_freq_names and not in token_index
        self.assertIn(("US", "universal"), pruned_idx.high_freq_names)
        self.assertNotIn(("US", "universal"), pruned_idx.token_index)

    def test_cap_loss_breakdown(self):
        # When cap is 1, but query generates multiple true matches, loss is attributed to cap
        target_a = {"entity_id": "S2-A", "business_name": "Omega Robotics Inc", "business_address": "", "country": "US"}
        target_b = {"entity_id": "S2-B", "business_name": "Omega Robotics Solutions", "business_address": "", "country": "US"}
        idx = CandidateIndex(source_name="Source2").build_from_records([target_a, target_b])

        queries = [
            {"entity_id": "S1-1", "business_name": "Omega Robotics Corp", "business_address": "", "country": "US"},
        ]
        gt_map = {
            "S1-1": {"S2-A", "S2-B"},
        }
        capped_gen = CandidateGenerator(idx, max_candidates_per_entity=1)
        stats = evaluate_blocking_recall(queries, capped_gen, gt_map)
        self.assertEqual(stats["total_gt_matches"], 2)
        self.assertEqual(stats["retrieved_gt_matches"], 1)
        self.assertEqual(stats["gt_lost_to_cap"], 1)
        self.assertEqual(stats["gt_lost_to_blocking"], 0)
        self.assertEqual(stats["uncapped_recall_pct"], 100.0)
        self.assertEqual(stats["candidate_recall_pct"], 50.0)

    def test_sorted_name_blocking(self):
        # S1 has "Kumar Rajesh Trading" and S2 has "Rajesh Kumar Trading"
        target = {
            "entity_id": "S2-020",
            "business_name": "Rajesh Kumar Trading",
            "business_address": "",
            "country": "India",
        }
        idx = CandidateIndex(source_name="Source2").build_from_records([target])
        gen = CandidateGenerator(idx, max_candidates_per_entity=10)
        query = {
            "entity_id": "S1-020",
            "business_name": "Kumar Rajesh Trading Co",
            "business_address": "",
            "country": "India",
        }
        cands = gen.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-020", cand_ids)
        match = next(c for c in cands if c["candidate_entity_id"] == "S2-020")
        self.assertIn("sorted_name", match["blocking_methods"])

    def test_name_bigram_blocking(self):
        # Two common words that would otherwise be pruned, but bigram matches
        target = {
            "entity_id": "S2-021",
            "business_name": "Apex Global Healthcare Enterprises",
            "business_address": "",
            "country": "US",
        }
        idx = CandidateIndex(source_name="Source2", max_name_df=1).build_from_records([target])
        gen = CandidateGenerator(idx, max_candidates_per_entity=10)
        query = {
            "entity_id": "S1-021",
            "business_name": "Apex Global Solutions LLC",
            "business_address": "",
            "country": "US",
        }
        cands = gen.generate_candidates(query)
        cand_ids = [c["candidate_entity_id"] for c in cands]
        self.assertIn("S2-021", cand_ids)
        match = next(c for c in cands if c["candidate_entity_id"] == "S2-021")
        self.assertIn("name_bigram", match["blocking_methods"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


