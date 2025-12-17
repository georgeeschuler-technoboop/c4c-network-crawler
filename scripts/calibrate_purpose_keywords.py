"""
Purpose Keyword Calibration Script

ONE-TIME calibration tool to extract actual vocabulary from grants_detail.csv.
This is NOT runtime code — just a developer tool to seed keyword dictionaries.

Usage:
    python calibrate_purpose_keywords.py path/to/grants_detail.csv

Output:
    - Top 50 single words (4+ chars)
    - Top 50 two-word phrases
    - Sample grants for each top term (for context)
"""

import pandas as pd
import re
import sys
from collections import Counter
from pathlib import Path

# =============================================================================
# Normalization (same logic used in Insight Engine)
# =============================================================================

def normalize_purpose_text(s):
    """Normalize purpose text for keyword extraction."""
    if not s or pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
# Stopwords (common words to filter out)
# =============================================================================

STOPWORDS = {
    "the", "and", "for", "to", "of", "in", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "with", "from", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "over",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "than", "too", "very", "just", "also",
    "now", "its", "their", "our", "your", "which", "who", "whom", "what",
    "support", "program", "project", "grant", "fund", "funds", "funding",
    "organization", "organizations", "inc", "foundation", "provide", "provides",
    "work", "activities", "efforts", "initiative", "initiatives",
}


# =============================================================================
# Main Extraction
# =============================================================================

def extract_keywords(df: pd.DataFrame, purpose_col: str) -> dict:
    """Extract frequent words and phrases from purpose column."""
    
    # Normalize all purpose text
    df = df.copy()
    df["purpose_norm"] = df[purpose_col].apply(normalize_purpose_text)
    
    # Count single words (4+ chars, not stopwords)
    words = Counter()
    for text in df["purpose_norm"]:
        for w in text.split():
            if len(w) >= 4 and w not in STOPWORDS:
                words[w] += 1
    
    # Count two-word phrases
    phrases = Counter()
    for text in df["purpose_norm"]:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            # Skip if both are stopwords
            if w1 in STOPWORDS and w2 in STOPWORDS:
                continue
            phrases[f"{w1} {w2}"] += 1
    
    # Count three-word phrases (bonus)
    trigrams = Counter()
    for text in df["purpose_norm"]:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            trigrams[f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"] += 1
    
    return {
        "words": words,
        "phrases": phrases,
        "trigrams": trigrams,
        "df": df,
    }


def get_sample_grants(df: pd.DataFrame, term: str, n: int = 3) -> list:
    """Get sample grants containing a term."""
    mask = df["purpose_norm"].str.contains(term, na=False)
    samples = df[mask].head(n)
    return samples["purpose_norm"].tolist()


def print_results(results: dict, top_n: int = 50):
    """Print calibration results."""
    words = results["words"]
    phrases = results["phrases"]
    trigrams = results["trigrams"]
    df = results["df"]
    
    print("\n" + "=" * 70)
    print("PURPOSE KEYWORD CALIBRATION RESULTS")
    print("=" * 70)
    
    print(f"\nTotal grants analyzed: {len(df):,}")
    print(f"Grants with purpose text: {len(df[df['purpose_norm'] != '']):,}")
    
    # Top single words
    print("\n" + "-" * 70)
    print(f"TOP {top_n} SINGLE WORDS (4+ chars, excluding stopwords)")
    print("-" * 70)
    print(f"{'Rank':<6}{'Word':<25}{'Count':<10}{'%':<8}")
    print("-" * 50)
    
    total = len(df)
    for i, (word, count) in enumerate(words.most_common(top_n), 1):
        pct = 100 * count / total
        print(f"{i:<6}{word:<25}{count:<10}{pct:.1f}%")
    
    # Top two-word phrases
    print("\n" + "-" * 70)
    print(f"TOP {top_n} TWO-WORD PHRASES")
    print("-" * 70)
    print(f"{'Rank':<6}{'Phrase':<35}{'Count':<10}{'%':<8}")
    print("-" * 60)
    
    for i, (phrase, count) in enumerate(phrases.most_common(top_n), 1):
        pct = 100 * count / total
        print(f"{i:<6}{phrase:<35}{count:<10}{pct:.1f}%")
    
    # Top three-word phrases (top 20 only)
    print("\n" + "-" * 70)
    print("TOP 20 THREE-WORD PHRASES")
    print("-" * 70)
    print(f"{'Rank':<6}{'Phrase':<45}{'Count':<10}{'%':<8}")
    print("-" * 70)
    
    for i, (phrase, count) in enumerate(trigrams.most_common(20), 1):
        pct = 100 * count / total
        print(f"{i:<6}{phrase:<45}{count:<10}{pct:.1f}%")
    
    # Suggested category seeds
    print("\n" + "=" * 70)
    print("SUGGESTED CATEGORY SEEDS (based on frequency)")
    print("=" * 70)
    
    # Water-related
    water_terms = [w for w, c in words.most_common(200) 
                   if any(x in w for x in ["water", "lake", "river", "wetland", "aqua", "stream"])]
    water_phrases = [p for p, c in phrases.most_common(200)
                     if any(x in p for x in ["water", "lake", "river", "wetland"])]
    
    print("\n💧 WATER candidates:")
    print(f"  Words: {water_terms[:15]}")
    print(f"  Phrases: {water_phrases[:10]}")
    
    # Environment-related
    env_terms = [w for w, c in words.most_common(200)
                 if any(x in w for x in ["environment", "conserv", "habitat", "wildlife", "forest", "nature", "land", "ecolog"])]
    env_phrases = [p for p, c in phrases.most_common(200)
                   if any(x in p for x in ["environment", "conserv", "habitat", "wildlife", "land"])]
    
    print("\n🌲 ENVIRONMENT candidates:")
    print(f"  Words: {env_terms[:15]}")
    print(f"  Phrases: {env_phrases[:10]}")
    
    # Climate-related
    climate_terms = [w for w, c in words.most_common(200)
                     if any(x in w for x in ["climate", "carbon", "energy", "emission", "sustain", "renewable"])]
    climate_phrases = [p for p, c in phrases.most_common(200)
                       if any(x in p for x in ["climate", "carbon", "energy", "clean", "renewable"])]
    
    print("\n🌡️ CLIMATE candidates:")
    print(f"  Words: {climate_terms[:15]}")
    print(f"  Phrases: {climate_phrases[:10]}")
    
    # Education-related
    edu_terms = [w for w, c in words.most_common(200)
                 if any(x in w for x in ["educat", "research", "school", "university", "scholar", "learn", "train"])]
    edu_phrases = [p for p, c in phrases.most_common(200)
                   if any(x in p for x in ["education", "research", "school", "university"])]
    
    print("\n📚 EDUCATION candidates:")
    print(f"  Words: {edu_terms[:15]}")
    print(f"  Phrases: {edu_phrases[:10]}")
    
    # Community-related  
    comm_terms = [w for w, c in words.most_common(200)
                  if any(x in w for x in ["communit", "civic", "social", "health", "youth", "neighbor", "equit", "justice"])]
    comm_phrases = [p for p, c in phrases.most_common(200)
                    if any(x in p for x in ["community", "civic", "public", "social"])]
    
    print("\n👥 COMMUNITY candidates:")
    print(f"  Words: {comm_terms[:15]}")
    print(f"  Phrases: {comm_phrases[:10]}")
    
    # General support
    support_phrases = [p for p, c in phrases.most_common(200)
                       if any(x in p for x in ["general", "operating", "core support", "unrestricted"])]
    
    print("\n🎯 GENERAL SUPPORT candidates:")
    print(f"  Phrases: {support_phrases[:10]}")


def export_to_csv(results: dict, output_path: str):
    """Export results to CSV for easier review."""
    words = results["words"]
    phrases = results["phrases"]
    
    # Use /home/claude or current dir for output
    base_name = Path(output_path).stem
    output_dir = Path("/home/claude")
    
    # Words CSV
    words_df = pd.DataFrame([
        {"word": w, "count": c, "pct": 100*c/len(results["df"])}
        for w, c in words.most_common(200)
    ])
    words_path = output_dir / f"{base_name}_words.csv"
    words_df.to_csv(words_path, index=False)
    
    # Phrases CSV
    phrases_df = pd.DataFrame([
        {"phrase": p, "count": c, "pct": 100*c/len(results["df"])}
        for p, c in phrases.most_common(200)
    ])
    phrases_path = output_dir / f"{base_name}_phrases.csv"
    phrases_df.to_csv(phrases_path, index=False)
    
    print(f"\n✅ Exported to:")
    print(f"   {words_path}")
    print(f"   {phrases_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_purpose_keywords.py path/to/grants_detail.csv")
        print("\nThis script extracts frequent words and phrases from grant purpose text")
        print("to calibrate keyword dictionaries for the Insight Engine.")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Find purpose column
    purpose_col = None
    for col in ["grant_purpose", "purpose", "raw_purpose", "grant_purpose_raw", "description"]:
        if col in df.columns:
            purpose_col = col
            break
    
    if not purpose_col:
        print(f"Error: No purpose column found. Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Using purpose column: '{purpose_col}'")
    
    # Extract keywords
    results = extract_keywords(df, purpose_col)
    
    # Print results
    print_results(results)
    
    # Export to CSV
    export_to_csv(results, str(csv_path))
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Review the top words and phrases above
2. Identify any missing terms for your categories
3. Update CORE_PURPOSE_CATEGORIES in insights_app.py with new terms
4. Re-run the Insight Engine to verify classification accuracy
    """)


if __name__ == "__main__":
    main()
