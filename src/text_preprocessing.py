#!/usr/bin/env python3
"""
Amazon ML Hackathon: Business Entity Resolution Challenge
Stage 3 — Multilingual Text Preprocessing Module

Provides high-performance, memory-efficient text preprocessing for business
entity resolution across US, India, and France:
- Unicode NFKC normalization
- Safe punctuation normalization (preserving non-Latin scripts and combining marks)
- Latin accent stripping while preserving Devanagari nuktas/matras/chandrabindu
- Multi-tier corporate legal suffix extraction and normalization (US, India, France)
- Address standardization and abbreviation normalization (US, India, France)
- Safe handling of missing addresses (empty addresses never become shared matching keys)
- Dual representation: Conservative (precision-preserving) and Normalized (recall-oriented)
- Idempotent and reproducible transformations
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any

# ==============================================================================
# 1. TRANSLATION TABLES & PATTERNS (PRECOMPUTED FOR HIGH SPEED)
# ==============================================================================

def _build_punctuation_translation_table():
    """
    Builds a fast translate table for punctuation and symbols.
    Maps Unicode punctuation ('P') and symbol ('S') characters to spaces,
    except ampersand ('&') which is handled explicitly to preserve 'and'.
    CRITICAL: Preserves all letters ('L'), numbers ('N'), and combining marks ('M').
    Combining marks ('M') are essential for Devanagari vowel signs (matras), viramas,
    anusvara, and nuktas.
    """
    table = {}
    for cp in range(0x10000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if ch == '&':
            continue
        if cat.startswith(('P', 'S')):
            table[cp] = ' '
    return table

_PUNCTUATION_TABLE = _build_punctuation_translation_table()

# Precompiled regexes
_WHITESPACE_RE = re.compile(r'\s+')
_AMPERSAND_RE = re.compile(r'&')

# Combining Diacritical Marks block (Latin/Greek/Cyrillic diacritics)
# U+0300 to U+036F: Combining Diacritical Marks
# U+1DC0 to U+1DFF: Combining Diacritical Marks Supplement
# U+20D0 to U+20FF: Combining Diacritical Marks for Symbols
# U+FE20 to U+FE2F: Combining Half Marks
_LATIN_DIACRITICS_RE = re.compile(r'[\u0300-\u036f\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]')

# ==============================================================================
# 2. CORPORATE LEGAL ENTITY TERMS (US, INDIA, FRANCE)
# ==============================================================================

_STOPWORDS = {'the', 'a', 'an', 'and', 'of', 'for', 'le', 'la', 'les', 'l', 'd', 'de'}

CORPORATE_SUFFIXES_EN = [
    # Multi-word
    'private limited company',
    'private limited',
    'private ltd',
    'pvt limited',
    'pvt ltd',
    'limited liability partnership',
    'limited liability company',
    'limited partnership',
    'professional corporation',
    'sole proprietorship',
    'public limited company',
    'public limited',
    'co ltd',
    'company limited',
    'pvt co',
    # Single-word
    'incorporated',
    'corporation',
    'associates',
    'cooperative',
    'proprietor',
    'proprietorship',
    'holding',
    'holdings',
    'company',
    'limited',
    'private',
    'group',
    'assoc',
    'pllc',
    'llc',
    'llp',
    'inc',
    'corp',
    'ltd',
    'pvt',
    'co',
    'lp',
    'pc',
]

CORPORATE_SUFFIXES_FR = [
    # Multi-word & abbreviations
    'societe a responsabilite limitee',
    'societe par actions simplifiee unipersonnelle',
    'societe par actions simplifiee',
    'societe en commandite par actions',
    'societe en commandite simple',
    'societe en nom collectif',
    'societe civile immobiliere',
    'societe civile professionnelle',
    'societe anonyme',
    'groupement d interet economique',
    'entreprise unipersonnelle a responsabilite limitee',
    # Common French abbreviations
    'sarl',
    'sasu',
    'sas',
    'eurl',
    'sci',
    'snc',
    'gie',
    'sca',
    'scs',
    'scp',
    'sa',
    'ste',
]

CORPORATE_SUFFIXES_HI = [
    'प्राइवेट लिमिटेड',
    'प्राइवेट लि',
    'प्रा लि',
    'प्रा. लि.',
    'लिमिटेड',
    'लि.',
    'लि',
    'कंपनी',
    'कॉर्पोरेशन',
]

# Combined list of all suffixes, sorted by descending word count and length
ALL_CORPORATE_SUFFIXES = sorted(
    set(CORPORATE_SUFFIXES_EN + CORPORATE_SUFFIXES_FR + CORPORATE_SUFFIXES_HI),
    key=lambda s: (len(s.split()), len(s)),
    reverse=True
)

# Map variations to canonical corporate suffix
CORPORATE_SUFFIX_CANONICAL = {
    'incorporated': 'inc',
    'inc': 'inc',
    'corporation': 'corp',
    'corp': 'corp',
    'limited liability company': 'llc',
    'llc': 'llc',
    'limited liability partnership': 'llp',
    'llp': 'llp',
    'limited': 'ltd',
    'ltd': 'ltd',
    'private limited company': 'pvt ltd',
    'private limited': 'pvt ltd',
    'private ltd': 'pvt ltd',
    'pvt limited': 'pvt ltd',
    'pvt ltd': 'pvt ltd',
    'private': 'pvt ltd',
    'pvt': 'pvt ltd',
    'company': 'co',
    'co': 'co',
    'co ltd': 'ltd',
    'company limited': 'ltd',
    'sarl': 'sarl',
    'societe a responsabilite limitee': 'sarl',
    'sas': 'sas',
    'societe par actions simplifiee': 'sas',
    'sasu': 'sasu',
    'sa': 'sa',
    'societe anonyme': 'sa',
    'sci': 'sci',
    'eurl': 'eurl',
    'ste': 'ste',
    'प्राइवेट लिमिटेड': 'pvt ltd',
    'प्रा लि': 'pvt ltd',
    'प्रा. लि.': 'pvt ltd',
    'लिमिटेड': 'ltd',
    'लि': 'ltd',
    'लि.': 'ltd',
    'कंपनी': 'co',
    'कॉर्पोरेशन': 'corp',
}

# ==============================================================================
# 3. ADDRESS ABBREVIATIONS & STANDARDIZATIONS
# ==============================================================================

# Standardized street and unit mappings
ADDRESS_ABBREVIATIONS = {
    # US & General English
    'street': 'st',
    'str': 'st',
    'avenue': 'ave',
    'av': 'ave',
    'boulevard': 'blvd',
    'boul': 'blvd',
    'road': 'rd',
    'drive': 'dr',
    'lane': 'ln',
    'court': 'ct',
    'circle': 'cir',
    'highway': 'hwy',
    'parkway': 'pkwy',
    'square': 'sq',
    'place': 'pl',
    'terrace': 'ter',
    'suite': 'ste',
    'apartment': 'apt',
    'building': 'bldg',
    'floor': 'fl',
    'unit': 'unit',
    # India specific
    'taluk': 'tq',
    'taluka': 'tq',
    'district': 'dist',
    'opposite': 'opp',
    'near': 'nr',
    'behind': 'bh',
    'sector': 'sec',
    'phase': 'ph',
    'nagar': 'nagar',
    'colony': 'colony',
    'marg': 'marg',
    # France specific
    'rue': 'r',
    'boulevard': 'bd',
    'bld': 'bd',
    'allee': 'all',
    'impasse': 'imp',
    'chemin': 'che',
    'route': 'rte',
}


# ==============================================================================
# 4. CORE ATOMIC PREPROCESSING FUNCTIONS
# ==============================================================================

def normalize_unicode_nfkc(text: Optional[str]) -> str:
    """
    Applies Unicode NFKC (Compatibility Decomposition, followed by Canonical Composition).
    Normalizes ligature forms, full-width characters, and compatibility glyphs into canonical
    Unicode representations while preserving Devanagari and Latin text.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def to_lower(text: Optional[str]) -> str:
    """
    Converts text to lowercase across Unicode scripts.
    Preserves non-cased scripts like Devanagari untouched.
    """
    if not text:
        return ""
    return text.lower()


def normalize_whitespace(text: Optional[str]) -> str:
    """
    Collapses all consecutive whitespace characters (spaces, tabs, newlines,
    non-breaking spaces U+00A0) into a single ASCII space and strips boundaries.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_punctuation(text: Optional[str], keep_ampersand: bool = False) -> str:
    """
    Safely normalizes punctuation:
    - Ampersand '&' is replaced with ' and ' (unless keep_ampersand=True).
    - Punctuation and symbols are replaced with space.
    - PRESERVES all letters ('L'), numbers ('N'), and combining marks ('M').
    Combining marks ('M') are strictly preserved to keep Hindi matras and vowel signs intact.
    """
    if not text:
        return ""
    t = text
    if not keep_ampersand:
        t = _AMPERSAND_RE.sub(" and ", t)
    t = t.translate(_PUNCTUATION_TABLE)
    return _WHITESPACE_RE.sub(" ", t).strip()


def strip_accents(text: Optional[str]) -> str:
    """
    Generates an accent-insensitive representation for Latin text (e.g. French 'École' -> 'Ecole'),
    while STRICTLY preserving Devanagari combining marks (nuktas, chandrabindu, matras).
    Uses NFD decomposition and only removes combining diacritical marks in the Latin range.
    """
    if not text:
        return ""
    # NFD decomposition splits characters into base character + combining mark
    nfd_text = unicodedata.normalize("NFD", text)
    # Strip only Latin/Greek combining diacritics (U+0300 to U+036F, etc.)
    stripped = _LATIN_DIACRITICS_RE.sub("", nfd_text)
    # Recompose to canonical NFC
    return unicodedata.normalize("NFC", stripped)


_LAST_SUFFIX_WORDS = {s.split()[-1] for s in ALL_CORPORATE_SUFFIXES}


# ==============================================================================
# 5. CORPORATE LEGAL ENTITY EXTRACTION
# ==============================================================================

def extract_corporate_suffix(name: Optional[str], country: Optional[str] = None) -> Tuple[str, str]:
    """
    Extracts and strips corporate legal suffixes (US, India, France) from the end of a business name.

    Returns:
        (core_name, corporate_suffix)
        where core_name is the name without the trailing suffix,
        and corporate_suffix is the extracted canonical legal term (or '' if none).

    Safety Guarantee:
        If stripping leaves an empty string or only stop words (e.g. name was literally
        'The Company' or 'Inc'), the original name is preserved to prevent creating empty
        or non-distinct matching keys.
    """
    if not name:
        return "", ""

    working_name = normalize_whitespace(name)
    tokens = working_name.split()
    if not tokens:
        return "", ""

    # Select candidate suffixes based on country if provided, else check all
    if country == "France":
        suffix_list = CORPORATE_SUFFIXES_FR + CORPORATE_SUFFIXES_EN
    elif country == "India":
        suffix_list = CORPORATE_SUFFIXES_HI + CORPORATE_SUFFIXES_EN
    else:
        suffix_list = ALL_CORPORATE_SUFFIXES

    curr_tokens = list(tokens)
    extracted = []

    # Loop to handle compound suffixes (e.g. 'Co Ltd' -> strips 'ltd' then 'co')
    while curr_tokens:
        matched = False
        lower_accent_stripped = [strip_accents(t).lower() for t in curr_tokens]
        # Fast O(1) early exit: if last token is not the end of any suffix, exit loop
        if lower_accent_stripped[-1] not in _LAST_SUFFIX_WORDS:
            break
        for suffix in suffix_list:
            suffix_words = suffix.split()
            k = len(suffix_words)
            if len(curr_tokens) >= k:
                candidate_suffix = " ".join(lower_accent_stripped[-k:])
                if candidate_suffix == suffix:
                    rem = curr_tokens[:-k]
                    # Safety check: remaining must have at least one non-stopword
                    non_stop = [t for t in rem if strip_accents(t).lower() not in _STOPWORDS]
                    if non_stop:
                        curr_tokens = rem
                        extracted.append(suffix)
                        matched = True
                        break
        if not matched:
            break

    if not extracted:
        return working_name, ""

    canonical_suffix = ""
    for s in extracted:
        if s in CORPORATE_SUFFIX_CANONICAL:
            canonical_suffix = CORPORATE_SUFFIX_CANONICAL[s]
            break

    return " ".join(curr_tokens), canonical_suffix


# ==============================================================================
# 6. BUSINESS NAME NORMALIZATION
# ==============================================================================

def normalize_business_name(
    name: Optional[str],
    country: Optional[str] = None,
    strip_suffix: bool = True
) -> Dict[str, Any]:
    """
    Normalizes a business name while retaining multiple representations:
    1. 'raw': The exact original input string.
    2. 'conservative': Cleaned, lowercased, NFKC normalized, whitespace trimmed,
       punctuation normalized, but retaining corporate suffixes and accents.
    3. 'normalized': Aggressively normalized for candidate generation & matching:
       accent-stripped, punctuation normalized, lowercased, with corporate suffixes removed.
    4. 'corporate_suffix': Canonical extracted corporate designator (or '').
    5. 'tokens': Clean list of distinctive name tokens.
    """
    if not name or not isinstance(name, str):
        return {
            "raw": "",
            "conservative": "",
            "normalized": "",
            "corporate_suffix": "",
            "tokens": [],
        }

    raw = name
    # 1. Unicode NFKC
    nfkc = normalize_unicode_nfkc(raw)
    # 2. Lowercase
    lowered = to_lower(nfkc)
    # 3. Punctuation normalization
    punct_clean = normalize_punctuation(lowered)
    # 4. Conservative representation
    conservative = normalize_whitespace(punct_clean)

    # 5. Extract corporate suffix
    if strip_suffix:
        core_name, extracted_suffix = extract_corporate_suffix(conservative, country=country)
    else:
        core_name, extracted_suffix = conservative, ""

    # 6. Accent-stripped representation (preserving non-Latin scripts)
    normalized = strip_accents(core_name)
    normalized = normalize_whitespace(normalized)

    # If stripping left empty string, fallback to conservative
    if not normalized:
        normalized = conservative

    tokens = [t for t in normalized.split() if t]

    return {
        "raw": raw,
        "conservative": conservative,
        "normalized": normalized,
        "corporate_suffix": extracted_suffix,
        "tokens": tokens,
    }


# ==============================================================================
# 7. BUSINESS ADDRESS NORMALIZATION
# ==============================================================================

def normalize_business_address(
    address: Optional[str],
    country: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalizes a business address safely:
    CRITICAL RULE:
    An empty or missing address remains strictly empty (""). It will NOT become
    a shared token or placeholder, preventing spurious false positive matches.

    Returns:
    1. 'raw': Original address.
    2. 'is_empty': True if missing/whitespace-only.
    3. 'conservative': NFKC, lowercased, punctuation normalized, trimmed.
    4. 'normalized': Standardized street/unit abbreviations, accent-stripped.
    5. 'tokens': List of distinctive address tokens.
    """
    if not address or not isinstance(address, str) or not address.strip():
        return {
            "raw": "" if address is None else address,
            "is_empty": True,
            "conservative": "",
            "normalized": "",
            "tokens": [],
        }

    raw = address
    nfkc = normalize_unicode_nfkc(raw)
    lowered = to_lower(nfkc)
    punct_clean = normalize_punctuation(lowered)
    conservative = normalize_whitespace(punct_clean)

    if not conservative:
        return {
            "raw": raw,
            "is_empty": True,
            "conservative": "",
            "normalized": "",
            "tokens": [],
        }

    # Standardize abbreviations on word boundaries
    words = conservative.split()
    standardized_words = []
    for w in words:
        std_w = ADDRESS_ABBREVIATIONS.get(w, w)
        standardized_words.append(std_w)

    standardized_address = " ".join(standardized_words)
    normalized = strip_accents(standardized_address)
    normalized = normalize_whitespace(normalized)
    tokens = [t for t in normalized.split() if t]

    return {
        "raw": raw,
        "is_empty": False,
        "conservative": conservative,
        "normalized": normalized,
        "tokens": tokens,
    }


# ==============================================================================
# 8. FULL RECORD PREPROCESSING HELPER
# ==============================================================================

def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocesses an entire entity record.
    Preserves entity_id without using it as a matching feature.
    """
    eid = record.get("entity_id", "")
    bname = record.get("business_name", "")
    baddr = record.get("business_address", "")
    country = record.get("country", "")

    name_dict = normalize_business_name(bname, country=country)
    addr_dict = normalize_business_address(baddr, country=country)

    return {
        "entity_id": eid,
        "country": country,
        "name_raw": name_dict["raw"],
        "name_conservative": name_dict["conservative"],
        "name_normalized": name_dict["normalized"],
        "name_corporate_suffix": name_dict["corporate_suffix"],
        "name_tokens": name_dict["tokens"],
        "addr_raw": addr_dict["raw"],
        "addr_is_empty": addr_dict["is_empty"],
        "addr_conservative": addr_dict["conservative"],
        "addr_normalized": addr_dict["normalized"],
        "addr_tokens": addr_dict["tokens"],
    }
