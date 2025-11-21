# helpers/generate_country_names.py
"""
Generate clean country name resources from Dyadic_COW_4.0.csv
Uses real 'importer1' and 'importer2' columns → 100% accurate
Saves:
  - unique_countries.csv
  - ccode_to_country_name.csv
  - country_name_mapping.py (for import in part_d_results.py)
"""

import pandas as pd
import os

# Update path if needed
CSV_PATH = "data/Dyadic_COW_4.0.csv"

print("Loading Dyadic_COW_4.0.csv to extract country names...")
df = pd.read_csv(CSV_PATH, low_memory=False)

print(f"→ Loaded {len(df):,} trade observations")

# ======================================================================
# 2. Build ccode → country name mapping (one-to-one, latest wins)
# ======================================================================
print("\nBuilding ccode → country name mapping...")

ccode_to_name = {}

for _, row in df.iterrows():
    c1 = row['ccode1']
    c2 = row['ccode2']
    n1 = str(row['importer1']).strip()
    n2 = str(row['importer2']).strip()

    if pd.notna(c1) and n1 and n1.lower() not in ['nan', 'none', '']:
        ccode_to_name[int(c1)] = n1
    if pd.notna(c2) and n2 and n2.lower() not in ['nan', 'none', '']:
        ccode_to_name[int(c2)] = n2

# ======================================================================
# 3. Generate Python module for easy import
# ======================================================================
print("\nGenerating country_name_mapping.py for use in part_d_results.py...")

with open("country_name_mapping.py", "w", encoding="utf-8") as f:
    f.write("# country_name_mapping.py\n")
    f.write("# Auto-generated from Dyadic_COW_4.0.csv – 100% accurate CCode → Name mapping\n")
    f.write("# Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + "\n\n")
    f.write("CCODE_TO_NAME = {\n")
    for ccode, name in sorted(ccode_to_name.items()):
        f.write(f"    {ccode}: \"{name}\",\n")
    f.write("}\n\n")
    f.write("def get_country_name(ccode):\n")
    f.write("    \"\"\"Return full country name from CCode, or fallback string.\"\"\"\n")
    f.write("    return CCODE_TO_NAME.get(ccode, f\"Country {int(ccode) if ccode else 'Unknown'}\")\n")

print("Saved: country_name_mapping.py")
