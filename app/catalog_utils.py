import re
import unicodedata
from rapidfuzz import fuzz, process

PLACEHOLDER_PATTERNS = [
    r"^\s*(sin datos|disponible|no usar|articulo.*prueba|registro disponible|descripcion|servicio de diseño)\s*$",
    r"^\s*(otros cargos|publicidad digital|servicios.*|honorarios|difusion|combo|set|kit|banner|araña publicitaria)\s*$",
]

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize_name(s: str) -> str:
    s = s.strip()
    s = s.replace("  ", " ")
    s = strip_accents(s)
    s = re.sub(r"[^a-zA-Z0-9\s\-\.\,\/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper().strip()

def is_placeholder(s: str) -> bool:
    s_up = s.strip().upper()
    for pat in PLACEHOLDER_PATTERNS:
        if re.match(pat, s_up, flags=re.IGNORECASE):
            return True
    return False

def fuzzy_best_match(query: str, candidates: list[str], limit: int = 5):
    # RapidFuzz WRatio is robust for mixed strings
    return process.extract(query, candidates, scorer=fuzz.WRatio, limit=limit)
