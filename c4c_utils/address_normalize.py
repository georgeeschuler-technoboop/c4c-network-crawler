# c4c_utils/address_normalize.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

US_STATE_RE = re.compile(r"\b([A-Z]{2})\b")
CA_PROV_RE = re.compile(r"\b(AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT)\b")

CITY_STATE_ZIP_RE = re.compile(
    r"^\s*(?P<city>[A-Za-z\.\'\-\s]+?)\s*,\s*(?P<admin1>[A-Z]{2})\s*(?P<postal>\d{5}(?:-\d{4})?)?\s*$"
)

@dataclass
class NormalizedAddress:
    raw: str = ""
    line1: str = ""
    city: str = ""
    admin1: str = ""   # US state or CA province code
    postal: str = ""
    country: str = ""  # "US" | "CA" | ""

def normalize_city_admin1(line: str) -> tuple[str, str, str]:
    """
    Extract city + admin1 + postal from 'City, ST 12345' (US) and similar.
    """
    m = CITY_STATE_ZIP_RE.match(line)
    if not m:
        return "", "", ""
    return (m.group("city").strip(), m.group("admin1").strip(), (m.group("postal") or "").strip())

def infer_country_from_admin1(admin1: str) -> str:
    if not admin1:
        return ""
    if admin1 in {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD",
                  "ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC",
                  "SD","TN","TX","UT","VA","VT","WA","WI","WV","WY","DC"}:
        return "US"
    if admin1 in {"AB","BC","MB","NB","NL","NS","NT","NU","ON","PE","QC","SK","YT"}:
        return "CA"
    return ""
