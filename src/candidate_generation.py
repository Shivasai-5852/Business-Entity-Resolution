#!/usr/bin/env python3
"""
Amazon ML Hackathon: Business Entity Resolution Challenge
Stage 4 — Multi-Pass Candidate Blocking and Generation Module

Provides scalable, memory-efficient candidate pair generation between Source 1
and target datasets (Source 2 and Source 3 independently):
1. Country-based partitioning (US, India, France, Unknown fallback)
2. Exact normalized & conservative name blocking
3. Distinctive name token inverted indexing (with Document Frequency thresholding)
4. Distinctive address token inverted indexing (excluding empty addresses & stop tokens)
5. Character n-gram prefix blocking (for typo & variation tolerance)
6. Multi-pass candidate merging, deduplication, and provenance tracking
7. Configurable candidate caps and priority ranking to balance recall vs candidates/entity
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional, Iterable

from text_preprocessing import (
    preprocess_record,
    normalize_business_name,
    normalize_business_address,
)

# Common address stop tokens to prevent giant uninformative address blocks
ADDRESS_STOP_TOKENS = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "circle", "cir", "highway", "hwy", "way", "parkway", "pkwy",
    "suite", "ste", "unit", "floor", "fl", "building", "bldg", "room", "rm",
    "near", "nr", "opp", "opposite", "behind", "bh", "city", "state", "town",
    "post", "box", "po", "and", "the", "of", "in", "at", "to", "for", "on",
    "north", "south", "east", "west", "central", "main", "new", "old", "first",
    "village", "nagar", "colony", "block", "sector", "phase", "plot", "shop",
}

# Common business name stop tokens
NAME_STOP_TOKENS = {
    "the", "a", "an", "and", "of", "for", "in", "at", "on", "to", "by", "with",
    "co", "corp", "inc", "ltd", "llc", "llp", "pvt", "limited", "company",
    "de", "la", "le", "les", "du", "des", "et", "en", "sur",
}


# Leading honorifics and organizational tags to clean for exact name matching
LEADING_PREFIX_RE = re.compile(
    r"^(mr|mrs|ms|dr|shri|smt|m/s|messrs|llp|pvt ltd)\b[\s\.\-\/]*",
    re.IGNORECASE
)


class CandidateIndex:
    """
    In-memory multi-pass blocking index for a target entity dataset (Source 2 or Source 3).
    Supports country-level partitioning, exact name lookup, distinctive token inverted index,
    address token index, character prefix index, consecutive token bigrams, and sorted name blocking.
    Features single-pass in-flight pruning for high-frequency terms to guarantee tight memory bounds.
    """

    def __init__(
        self,
        source_name: str = "Source2",
        max_name_df: int = 400,
        max_addr_df: int = 150,
        max_bigram_df: int = 200,
        max_block_size: int = 500,
        prefix_len: int = 6,
    ):
        self.source_name = source_name
        self.max_name_df = max_name_df
        self.max_addr_df = max_addr_df
        self.max_bigram_df = max_bigram_df
        self.max_block_size = max_block_size
        self.prefix_len = prefix_len

        # Blocking index tables: key -> list of candidate entity_ids
        self.exact_name_index = defaultdict(list)
        self.exact_cons_index = defaultdict(list)
        self.sorted_name_index = defaultdict(list)
        self.bigram_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.prefix_index = defaultdict(list)

        # High-frequency pruning sets to cap memory dynamically in a single streaming pass
        self.high_freq_names = set()
        self.high_freq_bigrams = set()
        self.high_freq_addrs = set()
        self.high_freq_prefixes = set()

        # Metadata
        self.total_records = 0
        self.country_record_counts = Counter()

    def add_record(self, record: Dict[str, Any]) -> None:
        """
        Indexes a single entity record in a single pass with immediate memory pruning.
        """
        if "name_normalized" not in record:
            p_rec = preprocess_record(record)
        else:
            p_rec = record

        self.total_records += 1
        eid = p_rec["entity_id"]
        country = p_rec.get("country", "") or "UNKNOWN"
        self.country_record_counts[country] += 1

        norm_name = p_rec["name_normalized"]
        cons_name = p_rec.get("name_conservative", "")

        # 1. Exact normalized name
        if norm_name:
            self.exact_name_index[(country, norm_name)].append(eid)
            # Leading honorific stripped variant (e.g. "Mr Global Traders" -> "global traders")
            clean_norm = LEADING_PREFIX_RE.sub("", norm_name).strip()
            if clean_norm and clean_norm != norm_name:
                self.exact_name_index[(country, clean_norm)].append(eid)

        # 2. Exact conservative name
        if cons_name and cons_name != norm_name:
            self.exact_cons_index[(country, cons_name)].append(eid)

        # 3. Sorted name tokens (word-order invariant blocking)
        toks = p_rec["name_tokens"]
        if len(toks) >= 2:
            sorted_key = (country, " ".join(sorted(toks)))
            lst = self.sorted_name_index[sorted_key]
            if len(lst) < self.max_block_size:
                lst.append(eid)

        # 4. Consecutive name token bigrams (rescues two-word names composed of common words)
        if len(toks) >= 2:
            for i in range(len(toks) - 1):
                bg = f"{toks[i]}_{toks[i+1]}"
                key = (country, bg)
                if key in self.high_freq_bigrams:
                    continue
                lst = self.bigram_index[key]
                lst.append(eid)
                if len(lst) > self.max_bigram_df:
                    del self.bigram_index[key]
                    self.high_freq_bigrams.add(key)

        # 5. Distinctive name unigram tokens with dynamic pruning
        for tok in toks:
            if tok not in NAME_STOP_TOKENS and len(tok) >= 3:
                key = (country, tok)
                if key in self.high_freq_names:
                    continue
                lst = self.token_index[key]
                lst.append(eid)
                if len(lst) > self.max_name_df:
                    del self.token_index[key]
                    self.high_freq_names.add(key)

        # 6. Distinctive address tokens (NEVER index empty addresses, include valid numeric tokens)
        if not p_rec.get("addr_is_empty", True):
            for tok in p_rec["addr_tokens"]:
                if tok not in ADDRESS_STOP_TOKENS and (len(tok) >= 4 or (tok.isdigit() and len(tok) >= 2)):
                    key = (country, tok)
                    if key in self.high_freq_addrs:
                        continue
                    lst = self.addr_token_index[key]
                    lst.append(eid)
                    if len(lst) > self.max_addr_df:
                        del self.addr_token_index[key]
                        self.high_freq_addrs.add(key)

        # 7. Character prefix 6-gram with dynamic pruning
        if norm_name:
            n_compact = norm_name.replace(" ", "")
            if len(n_compact) >= self.prefix_len:
                prefix_key = (country, n_compact[:self.prefix_len])
                if prefix_key not in self.high_freq_prefixes:
                    lst = self.prefix_index[prefix_key]
                    lst.append(eid)
                    if len(lst) > self.max_block_size:
                        del self.prefix_index[prefix_key]
                        self.high_freq_prefixes.add(prefix_key)

    def build_from_records(self, records: Iterable[Dict[str, Any]]) -> "CandidateIndex":
        """
        Builds blocking index structures from an iterable of preprocessed entity records.
        """
        for rec in records:
            self.add_record(rec)
        return self

    def build_from_tsv(
        self,
        tsv_path: str,
        country: Optional[str] = None,
        chunksize: int = 100000,
        max_rows: Optional[int] = None,
    ) -> "CandidateIndex":
        """
        Streams records directly from a TSV file in chunks, optionally filtering by country.
        Avoids accumulating raw or intermediate records in memory.
        """
        import pandas as pd
        rows_indexed = 0
        for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=chunksize, dtype=str):
            if country is not None:
                chunk = chunk[chunk["country"] == country]
            if len(chunk) == 0:
                continue

            for rec in chunk.to_dict("records"):
                self.add_record(rec)
                rows_indexed += 1
                if max_rows and rows_indexed >= max_rows:
                    return self
        return self


class CandidateGenerator:
    """
    Multi-pass candidate generator that queries CandidateIndex to produce
    prioritized, deduplicated candidate matches for query Source 1 entities.
    """

    def __init__(
        self,
        index: CandidateIndex,
        max_candidates_per_entity: int = 50,
        enable_exact_name: bool = True,
        enable_exact_conservative: bool = True,
        enable_sorted_name: bool = True,
        enable_name_bigrams: bool = True,
        enable_name_tokens: bool = True,
        enable_addr_tokens: bool = True,
        enable_prefix_ngram: bool = True,
        max_tokens_to_query: int = 3,
        max_block_cap: int = 500,
    ):
        self.index = index
        self.max_candidates = max_candidates_per_entity
        self.enable_exact_name = enable_exact_name
        self.enable_exact_conservative = enable_exact_conservative
        self.enable_sorted_name = enable_sorted_name
        self.enable_name_bigrams = enable_name_bigrams
        self.enable_name_tokens = enable_name_tokens
        self.enable_addr_tokens = enable_addr_tokens
        self.enable_prefix_ngram = enable_prefix_ngram
        self.max_tokens_to_query = max_tokens_to_query
        self.max_block_cap = max_block_cap

    def generate_candidates_with_meta(
        self, query_record: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Set[str], int]:
        """
        Generates candidates returning (capped_results, all_generated_candidate_ids, raw_candidate_count).
        """
        if "name_normalized" not in query_record:
            q_rec = preprocess_record(query_record)
        else:
            q_rec = query_record

        s1_id = q_rec["entity_id"]
        country = q_rec.get("country", "") or "UNKNOWN"

        candidates = defaultdict(lambda: {"score": 0, "passes": set()})

        # ----------------------------------------------------------------------
        # Pass 1: Exact Normalized Name (including prefix-stripped variant)
        # ----------------------------------------------------------------------
        norm_name = q_rec["name_normalized"]
        if self.enable_exact_name and norm_name:
            block = self.index.exact_name_index.get((country, norm_name), [])
            if 0 < len(block) <= self.max_block_cap:
                for cid in block:
                    candidates[cid]["score"] += 100
                    candidates[cid]["passes"].add("exact_name")

            # Check leading honorific stripped variant (e.g. "Mr Global Traders" -> "global traders")
            clean_norm = LEADING_PREFIX_RE.sub("", norm_name).strip()
            if clean_norm and clean_norm != norm_name:
                block_clean = self.index.exact_name_index.get((country, clean_norm), [])
                if 0 < len(block_clean) <= self.max_block_cap:
                    for cid in block_clean:
                        candidates[cid]["score"] += 95
                        candidates[cid]["passes"].add("exact_name")

        # ----------------------------------------------------------------------
        # Pass 2: Exact Conservative Name
        # ----------------------------------------------------------------------
        cons_name = q_rec.get("name_conservative", "")
        if self.enable_exact_conservative and cons_name and cons_name != norm_name:
            block = self.index.exact_cons_index.get((country, cons_name), [])
            if 0 < len(block) <= self.max_block_cap:
                for cid in block:
                    candidates[cid]["score"] += 80
                    candidates[cid]["passes"].add("exact_conservative")

        # ----------------------------------------------------------------------
        # Pass 3: Sorted Name Tokens (word-order invariant blocking)
        # ----------------------------------------------------------------------
        toks = q_rec["name_tokens"]
        if self.enable_sorted_name and len(toks) >= 2:
            sorted_key = (country, " ".join(sorted(toks)))
            block = self.index.sorted_name_index.get(sorted_key, [])
            if 0 < len(block) <= self.max_block_cap:
                for cid in block:
                    candidates[cid]["score"] += 70
                    candidates[cid]["passes"].add("sorted_name")

        # ----------------------------------------------------------------------
        # Pass 4: Consecutive Name Token Bigrams
        # ----------------------------------------------------------------------
        if self.enable_name_bigrams and len(toks) >= 2:
            for i in range(min(3, len(toks) - 1)):
                bg = f"{toks[i]}_{toks[i+1]}"
                key = (country, bg)
                block = self.index.bigram_index.get(key, [])
                if 0 < len(block) <= self.max_block_cap:
                    df = len(block)
                    bigram_weight = max(25, int(65 - math.log2(max(2, df))))
                    for cid in block:
                        candidates[cid]["score"] += bigram_weight
                        candidates[cid]["passes"].add("name_bigram")

        # ----------------------------------------------------------------------
        # Pass 5: Distinctive Name Unigram Tokens (Rarest tokens by DF)
        # ----------------------------------------------------------------------
        if self.enable_name_tokens and toks:
            valid_toks = [
                t for t in toks
                if t not in NAME_STOP_TOKENS and len(t) >= 3
                and (country, t) in self.index.token_index
            ]
            valid_toks.sort(key=lambda t: len(self.index.token_index.get((country, t), [])))
            for tok in valid_toks[:self.max_tokens_to_query]:
                block = self.index.token_index.get((country, tok), [])
                if 0 < len(block) <= self.max_block_cap:
                    df = len(block)
                    token_weight = max(12, int(35 - math.log2(max(2, df))))
                    for cid in block:
                        candidates[cid]["score"] += token_weight
                        candidates[cid]["passes"].add("name_token")

        # ----------------------------------------------------------------------
        # Pass 6: Distinctive Address Tokens (Priority numeric tokens + rarest words)
        # ----------------------------------------------------------------------
        if self.enable_addr_tokens and not q_rec.get("addr_is_empty", True):
            # Prioritize house/building numeric tokens (which have tiny block sizes)
            num_toks = [
                t for t in q_rec["addr_tokens"]
                if t.isdigit() and len(t) >= 2
                and (country, t) in self.index.addr_token_index
            ]
            # Word tokens sorted by ascending DF
            word_toks = [
                t for t in q_rec["addr_tokens"]
                if not t.isdigit() and t not in ADDRESS_STOP_TOKENS and len(t) >= 4
                and (country, t) in self.index.addr_token_index
            ]
            word_toks.sort(key=lambda t: len(self.index.addr_token_index.get((country, t), [])))

            # Query up to 2 numeric tokens AND up to 3 rarest word tokens
            addrs_to_query = num_toks[:2] + word_toks[:self.max_tokens_to_query]
            for tok in addrs_to_query:
                block = self.index.addr_token_index.get((country, tok), [])
                if 0 < len(block) <= self.max_block_cap:
                    df = len(block)
                    addr_weight = max(15, int(40 - math.log2(max(2, df))))
                    for cid in block:
                        candidates[cid]["score"] += addr_weight
                        candidates[cid]["passes"].add("addr_token")

        # ----------------------------------------------------------------------
        # Pass 7: Character Prefix 6-gram
        # ----------------------------------------------------------------------
        if self.enable_prefix_ngram and norm_name:
            n_compact = norm_name.replace(" ", "")
            if len(n_compact) >= self.index.prefix_len:
                prefix_key = (country, n_compact[:self.index.prefix_len])
                block = self.index.prefix_index.get(prefix_key, [])
                if 0 < len(block) <= self.max_block_cap:
                    for cid in block:
                        candidates[cid]["score"] += 8
                        candidates[cid]["passes"].add("prefix_6gram")

        all_generated_ids = set(candidates.keys())
        raw_count = len(candidates)

        # Multi-tier tie-breaking: prefer exact_name > bigram/sorted > name_token > addr_token
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: (
                item[1]["score"],
                len(item[1]["passes"]),
                1 if "exact_name" in item[1]["passes"] else 0,
                1 if "name_bigram" in item[1]["passes"] or "sorted_name" in item[1]["passes"] else 0,
                1 if "name_token" in item[1]["passes"] else 0,
                1 if "addr_token" in item[1]["passes"] else 0,
                item[0]
            ),
            reverse=True,
        )

        capped_candidates = sorted_candidates[:self.max_candidates]

        results = []
        for cid, data in capped_candidates:
            results.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "candidate_source": self.index.source_name,
                "blocking_methods": sorted(list(data["passes"])),
                "score": data["score"],
                "raw_candidate_count": raw_count,
            })

        return results, all_generated_ids, raw_count


    def generate_candidates(self, query_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates prioritized candidate matches for a single query entity record.
        """
        results, _, _ = self.generate_candidates_with_meta(query_record)
        return results


def evaluate_blocking_recall(
    query_records: List[Dict[str, Any]],
    candidate_generator: CandidateGenerator,
    ground_truth_map: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Evaluates candidate generation recall, candidate count distribution (both before
    and after candidate caps), and detailed loss breakdown against ground truth.
    """
    total_gt_matches = 0
    retrieved_gt_matches = 0
    gt_lost_to_blocking = 0
    gt_lost_to_cap = 0
    entities_with_candidates = 0
    entities_with_all_matches = 0
    entities_with_zero_matches = 0

    capped_candidate_counts = []
    raw_candidate_counts = []
    pass_retrieval_counts = Counter()
    country_recall = defaultdict(lambda: {"total": 0, "retrieved": 0, "lost_to_cap": 0})
    cardinality_recall = defaultdict(lambda: {"total": 0, "retrieved": 0})

    for rec in query_records:
        s1_id = rec["entity_id"]
        country = rec.get("country", "") or "UNKNOWN"
        true_matches = ground_truth_map.get(s1_id, set())
        n_true = len(true_matches)
        total_gt_matches += n_true

        if n_true == 0:
            entities_with_zero_matches += 1

        country_recall[country]["total"] += n_true
        cardinality_recall[n_true]["total"] += n_true

        # Generate candidates with metadata
        cands, all_generated_ids, raw_cands_count = candidate_generator.generate_candidates_with_meta(rec)
        n_capped = len(cands)
        capped_candidate_counts.append(n_capped)
        raw_candidate_counts.append(raw_cands_count)

        if n_capped > 0:
            entities_with_candidates += 1

        cand_ids = {c["candidate_entity_id"] for c in cands}
        matched_in_capped = true_matches & cand_ids
        matched_in_raw = true_matches & all_generated_ids

        retrieved_count = len(matched_in_capped)
        lost_to_cap_count = len(matched_in_raw) - retrieved_count
        lost_to_blocking_count = n_true - len(matched_in_raw)

        retrieved_gt_matches += retrieved_count
        gt_lost_to_cap += lost_to_cap_count
        gt_lost_to_blocking += lost_to_blocking_count

        country_recall[country]["retrieved"] += retrieved_count
        country_recall[country]["lost_to_cap"] += lost_to_cap_count
        cardinality_recall[n_true]["retrieved"] += retrieved_count

        if n_true > 0:
            if retrieved_count == n_true:
                entities_with_all_matches += 1

        # Track which passes contributed to finding true matches
        for c in cands:
            cid = c["candidate_entity_id"]
            if cid in true_matches:
                for p in c["blocking_methods"]:
                    pass_retrieval_counts[p] += 1

    n_entities = len(query_records)
    capped_candidate_counts.sort()
    raw_candidate_counts.sort()

    def get_percentile(data, p):
        if not data:
            return 0
        idx = int(p * len(data))
        return data[min(idx, len(data) - 1)]

    overall_recall = (retrieved_gt_matches / total_gt_matches * 100) if total_gt_matches > 0 else 0.0
    uncapped_recall = ((retrieved_gt_matches + gt_lost_to_cap) / total_gt_matches * 100) if total_gt_matches > 0 else 0.0

    return {
        "total_query_entities": n_entities,
        "total_gt_matches": total_gt_matches,
        "retrieved_gt_matches": retrieved_gt_matches,
        "gt_lost_to_blocking": gt_lost_to_blocking,
        "gt_lost_to_cap": gt_lost_to_cap,
        "candidate_recall_pct": round(overall_recall, 2),
        "uncapped_recall_pct": round(uncapped_recall, 2),
        "entities_with_at_least_one_candidate": entities_with_candidates,
        "pct_entities_with_candidates": round(entities_with_candidates / n_entities * 100, 2) if n_entities else 0.0,
        "entities_with_all_matches_retrieved": entities_with_all_matches,
        # Capped candidate stats
        "avg_candidates_per_entity": round(sum(capped_candidate_counts) / n_entities, 2) if n_entities else 0.0,
        "median_candidates": get_percentile(capped_candidate_counts, 0.50),
        "p95_candidates": get_percentile(capped_candidate_counts, 0.95),
        "max_candidates": capped_candidate_counts[-1] if capped_candidate_counts else 0,
        # Raw (before cap) candidate stats
        "avg_raw_candidates": round(sum(raw_candidate_counts) / n_entities, 2) if n_entities else 0.0,
        "median_raw_candidates": get_percentile(raw_candidate_counts, 0.50),
        "p95_raw_candidates": get_percentile(raw_candidate_counts, 0.95),
        "max_raw_candidates": raw_candidate_counts[-1] if raw_candidate_counts else 0,
        "pass_contributions": dict(pass_retrieval_counts),
        "country_recall": {
            c: {
                "total": d["total"],
                "retrieved": d["retrieved"],
                "lost_to_cap": d["lost_to_cap"],
                "recall_pct": round(d["retrieved"] / d["total"] * 100, 2) if d["total"] > 0 else 0.0,
            }
            for c, d in country_recall.items()
        },
        "cardinality_recall": {
            k: {
                "total": d["total"],
                "retrieved": d["retrieved"],
                "recall_pct": round(d["retrieved"] / d["total"] * 100, 2) if d["total"] > 0 else 0.0,
            }
            for k, d in sorted(cardinality_recall.items())
            if k > 0
        },
    }

