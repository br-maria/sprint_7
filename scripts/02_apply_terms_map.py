import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"
KB_DIR = BASE_DIR / "knowledge_base"
KB_DIR.mkdir(exist_ok=True, parents=True)

TERMS_MAP_FILE = BASE_DIR / "terms_map.json"

def load_terms_map():
    with open(TERMS_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_replacement_pattern(terms_map):
    # Сортируем по длине ключа, чтобы "Teenage Mutant Ninja Turtles"
    # заменился раньше, чем "Turtles"
    sorted_terms = sorted(terms_map.keys(), key=len, reverse=True)
    # Экранируем для regex
    escaped = [re.escape(t) for t in sorted_terms]
    pattern = re.compile("|".join(escaped))
    return pattern, terms_map

def replace_terms(text, pattern, terms_map):
    def _repl(match: re.Match):
        original = match.group(0)
        return terms_map.get(original, original)

    return pattern.sub(_repl, text)

def process_file(path, pattern, terms_map):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = replace_terms(content, pattern, terms_map)

    out_path = KB_DIR / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Processed: {path.name} -> {out_path}")

def main():
    terms_map = load_terms_map()
    pattern, terms_map = build_replacement_pattern(terms_map)

    md_files = sorted(RAW_DIR.glob("*.md")) + sorted(RAW_DIR.glob("*.txt"))

    for fp in md_files:
        process_file(fp, pattern, terms_map)

if __name__ == "__main__":
    main()
