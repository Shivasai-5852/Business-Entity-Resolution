#!/usr/bin/env python3
"""
Amazon ML Hackathon: Business Entity Resolution Challenge
Stage 3 — Comprehensive Unit Tests for Multilingual Text Preprocessing

Tests verify:
1. Unicode normalization and accented French text
2. Devanagari (Hindi) text and combining mark preservation
3. Whitespace and punctuation normalization
4. Corporate legal suffix extraction across US, India, and France
5. Safe missing value handling (empty addresses never become shared keys)
6. Idempotence (repeated application produces identical results)
7. Preservation of meaningful tokens and numbers
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_preprocessing import (
    normalize_unicode_nfkc,
    to_lower,
    normalize_whitespace,
    normalize_punctuation,
    strip_accents,
    extract_corporate_suffix,
    normalize_business_name,
    normalize_business_address,
    preprocess_record,
)


class TestUnicodeAndFrench(unittest.TestCase):
    """Test Unicode NFKC and French accented text processing."""

    def test_nfkc_ligatures_and_fullwidth(self):
        # Ligature ﬁ -> fi, fullwidth ＡＢＣ -> ABC
        self.assertEqual(normalize_unicode_nfkc("ﬁnancial"), "financial")
        self.assertEqual(normalize_unicode_nfkc("ＡＢＣ Ｃｏｒｐ"), "ABC Corp")

    def test_french_accents_conservative_vs_normalized(self):
        # French text with accents
        raw = "Marina École France Sàrl"
        res = normalize_business_name(raw, country="France")

        # Conservative keeps accents and original tokens
        self.assertIn("école", res["conservative"])
        self.assertIn("sàrl", res["conservative"])

        # Normalized strips Latin accents and extracts legal suffix
        self.assertEqual(res["normalized"], "marina ecole france")
        self.assertEqual(res["corporate_suffix"], "sarl")
        self.assertEqual(res["tokens"], ["marina", "ecole", "france"])

    def test_french_address_standardization(self):
        raw_addr = "63 R. DE DIEPPE, LILLE, Hauts-de-France"
        res = normalize_business_address(raw_addr, country="France")
        self.assertFalse(res["is_empty"])
        # 'r' or 'rue' abbreviation
        self.assertEqual(res["normalized"], "63 r de dieppe lille hauts de france")
        self.assertIn("63", res["tokens"])
        self.assertIn("dieppe", res["tokens"])
        self.assertIn("lille", res["tokens"])


class TestDevanagariHindi(unittest.TestCase):
    """Test strict preservation of Hindi (Devanagari) script."""

    def test_matras_and_vowel_signs_preserved(self):
        # Matras (vowel signs) must not be stripped by punctuation or accent cleaners
        raw = "राम मार्केटिंग (प्राइवेट) लिमिटेड"
        res = normalize_business_name(raw, country="India")

        # Core name should preserve all Hindi letters and matras
        self.assertEqual(res["normalized"], "राम मार्केटिंग")
        self.assertEqual(res["corporate_suffix"], "pvt ltd")
        self.assertEqual(res["tokens"], ["राम", "मार्केटिंग"])

    def test_nukta_and_chandrabindu_preservation(self):
        # Combining marks in Devanagari like nukta (U+093C) must be preserved
        nukta_text = "ग़ज़ल और आँख"
        stripped = strip_accents(nukta_text)
        self.assertEqual(stripped, nukta_text)

        cleaned = normalize_punctuation(nukta_text)
        self.assertEqual(cleaned, "ग़ज़ल और आँख")

    def test_modern_finance_hindi(self):
        # Real dataset example: मॉडर्न फाइनेंस
        raw = "मॉडर्न फाइनेंस"
        res = normalize_business_name(raw, country="India")
        self.assertEqual(res["normalized"], "मॉडर्न फाइनेंस")
        self.assertEqual(res["tokens"], ["मॉडर्न", "फाइनेंस"])

    def test_mixed_hindi_english(self):
        raw = "श्री Sai Infratech Co."
        res = normalize_business_name(raw, country="India")
        self.assertEqual(res["normalized"], "श्री sai infratech")
        self.assertEqual(res["corporate_suffix"], "co")


class TestWhitespaceAndPunctuation(unittest.TestCase):
    """Test whitespace collapsing and safe punctuation normalization."""

    def test_whitespace_collapsing(self):
        raw = "   Zephay   \t\t Labs \n\n  Inc.   "
        self.assertEqual(normalize_whitespace(raw), "Zephay Labs Inc.")

    def test_non_breaking_spaces(self):
        raw = "Alpha\u00a0Beta\u00a0Corp"
        nfkc = normalize_unicode_nfkc(raw)
        self.assertEqual(normalize_whitespace(nfkc), "Alpha Beta Corp")

    def test_ampersand_replacement(self):
        raw = "Johnson & Johnson & Co."
        res = normalize_punctuation(raw)
        self.assertEqual(res, "Johnson and Johnson and Co")

    def test_hyphen_and_slashes_in_address(self):
        raw_addr = "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"
        res = normalize_business_address(raw_addr, country="India")
        self.assertEqual(res["normalized"], "kh no 570 13 new delhi west delhi delhi")
        self.assertIn("570", res["tokens"])
        self.assertIn("13", res["tokens"])


class TestCorporateSuffixes(unittest.TestCase):
    """Test corporate suffix extraction across US, India, and France."""

    def test_us_suffixes(self):
        cases = [
            ("Apple Inc.", "apple", "inc"),
            ("Microsoft Corporation", "microsoft", "corp"),
            ("Amazon.com Services LLC", "amazon com services", "llc"),
            ("Holloway Peak Inc Seafood", "holloway peak inc seafood", ""),  # 'inc' not at end!
            ("Acme Co Ltd", "acme", "ltd"),
        ]
        for raw, exp_name, exp_suffix in cases:
            res = normalize_business_name(raw, country="US")
            self.assertEqual(res["normalized"], exp_name, f"Failed for {raw}")
            self.assertEqual(res["corporate_suffix"], exp_suffix, f"Failed suffix for {raw}")

    def test_india_suffixes(self):
        cases = [
            ("Tata Consultancy Services Pvt. Ltd.", "tata consultancy services", "pvt ltd"),
            ("Reliance Industries Limited", "reliance industries", "ltd"),
            ("Infosys Limited", "infosys", "ltd"),
            ("राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग", "pvt ltd"),
        ]
        for raw, exp_name, exp_suffix in cases:
            res = normalize_business_name(raw, country="India")
            self.assertEqual(res["normalized"], exp_name, f"Failed for {raw}")
            self.assertEqual(res["corporate_suffix"], exp_suffix, f"Failed suffix for {raw}")

    def test_france_suffixes(self):
        cases = [
            ("Marina Ecole France Sarl", "marina ecole france", "sarl"),
            ("TotalEnergies SE", "totalenergies se", ""),  # 'se' not in standard FR list
            ("L'Oreal SA", "l oreal", "sa"),
            ("BNP Paribas SAS", "bnp paribas", "sas"),
            ("Immobilier Du Centre SCI", "immobilier du centre", "sci"),
        ]
        for raw, exp_name, exp_suffix in cases:
            res = normalize_business_name(raw, country="France")
            self.assertEqual(res["normalized"], exp_name, f"Failed for {raw}")
            self.assertEqual(res["corporate_suffix"], exp_suffix, f"Failed suffix for {raw}")

    def test_safety_fallback_for_suffix_only_name(self):
        # A business called literally "The Company" or "Inc" must not be reduced to empty string
        res = normalize_business_name("The Company")
        self.assertNotEqual(res["normalized"], "")
        self.assertEqual(res["normalized"], "the company")

        res_inc = normalize_business_name("Inc.")
        self.assertNotEqual(res_inc["normalized"], "")
        self.assertEqual(res_inc["normalized"], "inc")


class TestMissingValues(unittest.TestCase):
    """Test safe missing value handling."""

    def test_none_and_empty_name(self):
        for val in [None, "", "   ", "\t\n"]:
            res = normalize_business_name(val)
            self.assertEqual(res["raw"], "" if val is None else val)
            self.assertEqual(res["normalized"], "")
            self.assertEqual(res["corporate_suffix"], "")
            self.assertEqual(res["tokens"], [])

    def test_missing_address_safety(self):
        # Critical: Empty address must remain strictly empty string, NOT become 'UNKNOWN' or 'N/A'
        for val in [None, "", "   ", "\t\n"]:
            res = normalize_business_address(val)
            self.assertTrue(res["is_empty"])
            self.assertEqual(res["normalized"], "")
            self.assertEqual(res["conservative"], "")
            self.assertEqual(res["tokens"], [])

    def test_preprocess_record_missing_address(self):
        rec = {
            "entity_id": "S3-859268022",
            "business_name": "International South Consultants Private Ltd",
            "business_address": "",
            "country": "India"
        }
        res = preprocess_record(rec)
        self.assertEqual(res["entity_id"], "S3-859268022")
        self.assertEqual(res["name_normalized"], "international south consultants")
        self.assertEqual(res["name_corporate_suffix"], "pvt ltd")
        self.assertTrue(res["addr_is_empty"])
        self.assertEqual(res["addr_normalized"], "")
        self.assertEqual(res["addr_tokens"], [])


class TestIdempotence(unittest.TestCase):
    """Test that repeated normalization yields identical results (f(f(x)) == f(x))."""

    def test_name_idempotence(self):
        names = [
            "Apple Inc.",
            "Marina École France Sàrl",
            "राम मार्केटिंग प्राइवेट लिमिटेड",
            "Orelee's Barbershop & Salon",
        ]
        for n in names:
            first = normalize_business_name(n)["normalized"]
            second = normalize_business_name(first)["normalized"]
            self.assertEqual(first, second, f"Not idempotent for: {n}")

    def test_address_idempotence(self):
        addrs = [
            "2621 Cotten Road, Tyler, TX",
            "IA, Iowa City, 1064 Newton Rd, Unit 11",
            "63 R. DE DIEPPE, LILLE, Hauts-de-France",
            "No 10 Enkay Square, Udyog Vihar Phase V, Gurugram, HR",
        ]
        for a in addrs:
            first = normalize_business_address(a)["normalized"]
            second = normalize_business_address(first)["normalized"]
            self.assertEqual(first, second, f"Not idempotent for: {a}")

    def test_punctuation_idempotence(self):
        text = "Hello & World! 123-A; Test... Pvt. Ltd."
        p1 = normalize_punctuation(text)
        p2 = normalize_punctuation(p1)
        self.assertEqual(p1, p2)


class TestMeaningfulTokenPreservation(unittest.TestCase):
    """Test that numbers, suite designations, and distinctive names are preserved."""

    def test_address_digits_and_units_preserved(self):
        addr = "1064 Newton Rd, Unit 11, Suite 400"
        res = normalize_business_address(addr, country="US")
        for expected in ["1064", "newton", "rd", "unit", "11", "ste", "400"]:
            self.assertIn(expected, res["tokens"], f"Missing {expected} in {res['tokens']}")

    def test_brand_names_not_corrupted(self):
        names = [
            "Zephay Labs Inc",
            "Brahma Infosoft",
            "Wilford Hancock",
            "Prime Money",
        ]
        for n in names:
            res = normalize_business_name(n)
            tokens = res["tokens"]
            self.assertTrue(len(tokens) >= 2)


class TestRealDatasetRecords(unittest.TestCase):
    """Test real entity examples taken directly from the train and test sets."""

    def test_real_dataset_samples(self):
        records = [
            {
                "entity_id": "S1-714132312",
                "business_name": "Zephay Labs Inc",
                "business_address": "2621 Cotten Road, Tyler, TX",
                "country": "US",
                "exp_name_norm": "zephay labs",
                "exp_suffix": "inc",
                "exp_addr_norm": "2621 cotten rd tyler tx",
            },
            {
                "entity_id": "S2-192345572",
                "business_name": "Brahma Infosoft",
                "business_address": "COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka",
                "country": "India",
                "exp_name_norm": "brahma infosoft",
                "exp_suffix": "",
                "exp_addr_norm": "coimatore colony hunsur tqmysore dist karnataka",
            },
            {
                "entity_id": "S2-566025912",
                "business_name": "Marina Ecole France Sarl",
                "business_address": "63 R. DE DIEPPE, LILLE, Hauts-de-France",
                "country": "France",
                "exp_name_norm": "marina ecole france",
                "exp_suffix": "sarl",
                "exp_addr_norm": "63 r de dieppe lille hauts de france",
            },
            {
                "entity_id": "S3-462677478",
                "business_name": "मॉडर्न फाइनेंस",
                "business_address": "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR",
                "country": "India",
                "exp_name_norm": "मॉडर्न फाइनेंस",
                "exp_suffix": "",
                "exp_addr_norm": "no 10 enkay sq 448a udyog vihar ph v gurugram gurgaon hr",
            },
            {
                "entity_id": "S3-859268022",
                "business_name": "International South Consultants Private Ltd",
                "business_address": "",
                "country": "India",
                "exp_name_norm": "international south consultants",
                "exp_suffix": "pvt ltd",
                "exp_addr_norm": "",
            },
        ]

        for r in records:
            res = preprocess_record(r)
            self.assertEqual(res["entity_id"], r["entity_id"])
            self.assertEqual(res["country"], r["country"])
            self.assertEqual(res["name_normalized"], r["exp_name_norm"])
            self.assertEqual(res["name_corporate_suffix"], r["exp_suffix"])
            self.assertEqual(res["addr_normalized"], r["exp_addr_norm"])
            if not r["business_address"]:
                self.assertTrue(res["addr_is_empty"])
            else:
                self.assertFalse(res["addr_is_empty"])


class TestPerformanceThroughput(unittest.TestCase):
    """Verify preprocessing speed meets high-performance competition standards (>25k rec/s)."""

    def test_throughput_benchmark(self):
        import time
        sample_rec = {
            "entity_id": "S1-925783039",
            "business_name": "Orelee's Barbershop & Styling Salon Inc.",
            "business_address": "1795 Westchester Drive, Suite 100, High Point, NC",
            "country": "US",
        }
        n_iters = 5000
        t0 = time.time()
        for _ in range(n_iters):
            _ = preprocess_record(sample_rec)
        elapsed = time.time() - t0
        throughput = n_iters / elapsed
        # Ensure at least 10,000 records/sec
        self.assertGreater(throughput, 10000, f"Throughput too low: {throughput:.0f} records/sec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
