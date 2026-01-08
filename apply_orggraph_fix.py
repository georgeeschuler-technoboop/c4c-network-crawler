#!/usr/bin/env python3
"""
OrgGraph v0.25.1 Patch Script
=============================
Fixes the fiscal_year mixed-type bug that causes st.dataframe() to hang.

Root Cause:
-----------
When DataFrames from different parsers (990-PF, 990, 990-EZ) are concatenated,
the fiscal_year column can have mixed types (some int, some str). PyArrow 
serialization fails on mixed-type columns, causing Streamlit's st.dataframe()
to hang indefinitely with a spinner.

Fixes Applied:
--------------
1. APP_VERSION bump: "0.25.0" → "0.25.1"
2. Type coercion in ensure_grants_detail_columns(): Force fiscal_year and 
   tax_year to string dtype before return
3. Deprecation fix: use_container_width=True → width="stretch" (2 locations)

Usage:
------
    python apply_orggraph_fix.py /path/to/funder_flow/app.py

The script creates a backup (.bak) before modifying.
"""

import sys
import re
from pathlib import Path


def apply_patch(filepath: str) -> tuple[bool, str]:
    """
    Apply the v0.25.1 patch to the OrgGraph app.py file.
    
    Returns:
        (success: bool, message: str)
    """
    path = Path(filepath)
    
    if not path.exists():
        return False, f"File not found: {filepath}"
    
    # Read original content
    content = path.read_text(encoding='utf-8')
    original_content = content
    
    changes_made = []
    
    # ==========================================================================
    # FIX 1: Update APP_VERSION
    # ==========================================================================
    old_version = 'APP_VERSION = "0.25.0"'
    new_version = 'APP_VERSION = "0.25.1"  # Fix fiscal_year Arrow serialization bug'
    
    if old_version in content:
        content = content.replace(old_version, new_version)
        changes_made.append("✅ FIX 1: Updated APP_VERSION to 0.25.1")
    elif 'APP_VERSION = "0.25.1"' in content:
        changes_made.append("⏭️ FIX 1: APP_VERSION already at 0.25.1 (skipped)")
    else:
        changes_made.append("⚠️ FIX 1: Could not find APP_VERSION line to update")
    
    # ==========================================================================
    # FIX 2: Add type coercion in ensure_grants_detail_columns()
    # ==========================================================================
    # We need to add the type coercion block before the final return statement
    # in the ensure_grants_detail_columns function
    
    type_coercion_block = '''    # CRITICAL: Force consistent dtypes for columns that cause Arrow serialization issues
    # fiscal_year and tax_year can be mixed int/str from different parsers
    # This prevents PyArrow from hanging when Streamlit tries to serialize DataFrames
    for col in ["fiscal_year", "tax_year"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    return df'''
    
    # Check if already patched
    if 'Force consistent dtypes for columns that cause Arrow serialization' in content:
        changes_made.append("⏭️ FIX 2: Type coercion already present (skipped)")
    else:
        # Find the ensure_grants_detail_columns function and its return statement
        # Pattern: look for the function and find the last "return df" inside it
        
        # First, find the function
        func_pattern = r'(def ensure_grants_detail_columns\([^)]*\)[^:]*:.*?)(    return df\n)'
        
        match = re.search(func_pattern, content, re.DOTALL)
        if match:
            # Replace the simple "return df" with our coercion block + return
            old_return = match.group(2)
            content = content.replace(
                match.group(0),
                match.group(1) + type_coercion_block + '\n'
            )
            changes_made.append("✅ FIX 2: Added type coercion for fiscal_year/tax_year")
        else:
            # Try alternative approach: just find and replace the pattern in context
            # Look for the specific pattern near the end of ensure_grants_detail_columns
            pattern2 = r'(for col in GRANTS_DETAIL_COLUMNS:\s+if col not in df\.columns:\s+df\[col\] = ""\s+)(return df)'
            match2 = re.search(pattern2, content)
            if match2:
                replacement = match2.group(1) + '\n' + type_coercion_block
                content = re.sub(pattern2, replacement, content)
                changes_made.append("✅ FIX 2: Added type coercion for fiscal_year/tax_year (alt method)")
            else:
                changes_made.append("⚠️ FIX 2: Could not locate insertion point for type coercion")
    
    # ==========================================================================
    # FIX 3: Replace deprecated use_container_width with width="stretch"
    # ==========================================================================
    deprecation_count = content.count('use_container_width=True')
    
    if deprecation_count > 0:
        content = content.replace('use_container_width=True', 'width="stretch"')
        changes_made.append(f"✅ FIX 3: Replaced {deprecation_count} instances of use_container_width=True")
    else:
        if 'width="stretch"' in content:
            changes_made.append("⏭️ FIX 3: Already using width=\"stretch\" (skipped)")
        else:
            changes_made.append("⚠️ FIX 3: No use_container_width=True found")
    
    # ==========================================================================
    # Write changes if any were made
    # ==========================================================================
    if content != original_content:
        # Create backup
        backup_path = path.with_suffix('.py.bak')
        backup_path.write_text(original_content, encoding='utf-8')
        
        # Write patched content
        path.write_text(content, encoding='utf-8')
        
        return True, f"Patch applied successfully!\n\nBackup saved to: {backup_path}\n\nChanges:\n" + "\n".join(changes_made)
    else:
        return True, "No changes needed (file already patched or patterns not found).\n\n" + "\n".join(changes_made)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide the path to app.py")
        print("Usage: python apply_orggraph_fix.py /path/to/funder_flow/app.py")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success, message = apply_patch(filepath)
    
    print("\n" + "=" * 60)
    print("OrgGraph v0.25.1 Patch")
    print("=" * 60 + "\n")
    print(message)
    print()
    
    if success:
        print("Next steps:")
        print("1. Test locally with: streamlit run app.py")
        print("2. Process a Ford Foundation XML to verify the fix")
        print("3. Deploy to Streamlit Cloud")
    else:
        print("Patch failed. Please apply changes manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
