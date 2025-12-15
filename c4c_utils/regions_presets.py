# regions_presets.py
"""
Region Presets for OrgGraph
===========================
Defines preset geographic regions for grant tagging.
Each region specifies US states and/or Canadian provinces.

Region tagging is optional enrichment applied after parsing.
"""

REGION_PRESETS = {
    "none": {
        "id": "none",
        "name": "None",
        "source": "preset",
        "include_us_states": [],
        "include_ca_provinces": [],
        "include_countries": [],
        "notes": "",
    },
    "new_england": {
        "id": "new_england",
        "name": "New England",
        "source": "preset",
        "include_us_states": ["ME", "NH", "VT", "MA", "RI", "CT"],
        "include_ca_provinces": [],
        "include_countries": ["US"],
        "notes": "",
    },
    "mid_atlantic": {
        "id": "mid_atlantic",
        "name": "Mid-Atlantic",
        "source": "preset",
        "include_us_states": ["NY", "NJ", "PA", "DE", "MD", "DC"],
        "include_ca_provinces": [],
        "include_countries": ["US"],
        "notes": "",
    },
    "great_lakes": {
        "id": "great_lakes",
        "name": "Great Lakes",
        "source": "preset",
        "include_us_states": ["MI", "OH", "MN", "WI", "IN", "IL", "NY"],
        "include_ca_provinces": ["ON", "QC"],
        "include_countries": ["US", "CA"],
        "notes": "Bi-national; primary project preset.",
    },
    "pacific_northwest": {
        "id": "pacific_northwest",
        "name": "Pacific Northwest",
        "source": "preset",
        "include_us_states": ["WA", "OR", "ID"],
        "include_ca_provinces": [],
        "include_countries": ["US"],
        "notes": "",
    },
    "california": {
        "id": "california",
        "name": "California",
        "source": "preset",
        "include_us_states": ["CA"],
        "include_ca_provinces": [],
        "include_countries": ["US"],
        "notes": "",
    },
}


# Complete lists for custom region builder
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY"
]

CA_PROVINCES = ["AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"]
