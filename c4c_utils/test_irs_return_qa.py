"""
test_irs_return_qa.py

Standalone test for the irs_return_qa module.
Run from the c4c_utils directory or adjust imports.

Usage:
    python test_irs_return_qa.py parse_log.json
    
Or just:
    python test_irs_return_qa.py
    (uses embedded sample data)
"""

import json
import sys

# If running from repo root, adjust path
try:
    from irs_return_qa import compute_confidence, ConfidenceResult
except ImportError:
    sys.path.insert(0, '.')
    from irs_return_qa import compute_confidence, ConfidenceResult


def test_with_file(filepath: str):
    """Test with a parse_log.json file."""
    print(f"\n📂 Loading: {filepath}\n")
    
    with open(filepath) as f:
        results = json.load(f)
    
    for r in results:
        test_single_result(r)


def test_single_result(r: dict):
    """Test confidence scoring on a single parse result."""
    diag = r.get("diagnostics", {})
    org_name = r.get("org_name", "Unknown")
    
    # Your parser uses "990-PF" implicitly - add form_type for scoring
    if "form_type_detected" not in diag:
        diag["form_type_detected"] = "990-PF"
    
    # Run confidence scoring
    conf = compute_confidence(diag)
    
    # Display results
    print("=" * 60)
    print(f"🏛️  {org_name}")
    print("=" * 60)
    print(f"Score:  {conf.score}/100")
    print(f"Grade:  {conf.grade.upper()}")
    print()
    
    print("✅ Reasons:")
    for r in conf.reasons:
        print(f"   • {r}")
    print()
    
    if conf.penalties:
        print("⚠️  Penalties:")
        for reason, points in conf.penalties:
            print(f"   • {reason} ({points})")
        print()
    
    # Show key diagnostic values used
    print("📊 Key diagnostics:")
    print(f"   • grants_3a_count: {diag.get('grants_3a_count', '—')}")
    print(f"   • grants_3a_total: ${diag.get('grants_3a_total', 0):,}")
    print(f"   • reported_total_3a: {diag.get('reported_total_3a', '—')}")
    print(f"   • grants_3b_count: {diag.get('grants_3b_count', '—')}")
    print(f"   • grants_3b_total: ${diag.get('grants_3b_total', 0):,}")
    print(f"   • reported_total_3b: {diag.get('reported_total_3b', '—')}")
    print()


def test_with_sample_data():
    """Test with embedded sample data (Joyce + Erb from your test run)."""
    print("\n📋 Using embedded sample data\n")
    
    sample_results = [
        {
            "org_name": "The Joyce Foundation",
            "diagnostics": {
                "parser_version": "2.6",
                "pages_processed": 66,
                "grants_3a_count": 515,
                "grants_3a_total": 55609321,
                "grants_3b_count": 148,
                "grants_3b_total": 20040898,
                "reported_total_3a": 55953961,  # From your parse_log
                "reported_total_3b": 20138189,
                "board_count": 23,
                "warnings": [],
                "errors": [],
            }
        },
        {
            "org_name": "THE FRED A & BARBARA M ERB FAMILY FOUNDATION",
            "diagnostics": {
                "parser_version": "2.6",
                "pages_processed": 38,
                "grants_3a_count": 149,
                "grants_3a_total": 17854055,
                "grants_3b_count": 92,
                "grants_3b_total": 13430305,
                "reported_total_3a": 17696127,
                "reported_total_3b": 13334070,
                "board_count": 8,
                "warnings": [],
                "errors": [],
            }
        },
        # Edge case: missing reported totals
        {
            "org_name": "Test Foundation (No Reported Totals)",
            "diagnostics": {
                "parser_version": "2.6",
                "pages_processed": 20,
                "grants_3a_count": 50,
                "grants_3a_total": 1000000,
                "grants_3b_count": 10,
                "grants_3b_total": 200000,
                # No reported_total_3a or reported_total_3b
                "warnings": [],
                "errors": [],
            }
        },
        # Edge case: zero grants but reported total > 0
        {
            "org_name": "Test Foundation (Zero Grants Bug)",
            "diagnostics": {
                "parser_version": "2.6",
                "pages_processed": 30,
                "grants_3a_count": 0,
                "grants_3a_total": 0,
                "reported_total_3a": 5000000,  # Parser missed grants!
                "warnings": [],
                "errors": [],
            }
        },
    ]
    
    for r in sample_results:
        test_single_result(r)


def main():
    if len(sys.argv) > 1:
        # Use provided file
        test_with_file(sys.argv[1])
    else:
        # Use sample data
        test_with_sample_data()
    
    print("\n✅ Test complete\n")


if __name__ == "__main__":
    main()
