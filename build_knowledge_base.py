import json
import re
from pathlib import Path
from typing import Dict, List

import requests
from faker import Faker
from slugify import slugify

import requests
import time

fake = Faker()

WIKI_API = "https://en.wikipedia.org/w/api.php"

SESSION = requests.Session()
SESSION.headers.update({
    # ВАЖНО: нормальный User-Agent. Можно поменять ссылку/почту на свои.
    "User-Agent": "QuantumForge-RAG-StudentBot/0.1 (https://github.com/SergeiLisovoi22/AI-ML; contact: sergeilisovoi22@gmail.com)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
})

# 30+ страниц (можете менять/расширять)
PAGES: List[str] = [
    "Darth Vader", "Luke Skywalker", "Leia Organa", "Han Solo", "Obi-Wan Kenobi",
    "Yoda", "Palpatine", "Boba Fett", "Chewbacca", "R2-D2", "C-3PO",
    "Jedi", "Sith", "The Force", "Lightsaber", "Death Star",
    "Millennium Falcon", "Tatooine", "Coruscant", "Naboo", "Endor",
    "Rebel Alliance", "Galactic Empire", "Clone Wars", "Order 66",
    "Star Wars (film)", "The Empire Strikes Back", "Return of the Jedi",
    "Star Wars: The Clone Wars", "Star Wars: Rebels", "Mandalorian",
    "Stormtrooper", "Droid", "Hyperspace", "Star Destroyer"
]

# Ключевые термины, которые важно заменить почти в любом тексте
CORE_TERMS = [
    "Star Wars", "Jedi", "Sith", "The Force", "lightsaber", "Death Star",
    "Galactic Empire", "Rebel Alliance", "stormtrooper", "droid", "hyperspace"
]

def wiki_extract(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": title,
    }

    # небольшой retry на случай 429/503
    for attempt in range(5):
        r = SESSION.get(WIKI_API, params=params, timeout=30)

        # Если rate-limit
        if r.status_code in (429, 503):
            time.sleep(1.5 * (attempt + 1))
            continue

        r.raise_for_status()
        data = r.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "") or ""
        return text.strip()

    raise RuntimeError(f"Failed to fetch '{title}' after retries (last status={r.status_code}).")

def make_terms_map(pages: List[str]) -> Dict[str, str]:
    """
    Создаём словарь замен. Для имён используем faker, для некоторых терминов — вручную.
    """
    mapping: Dict[str, str] = {}

    # Ручные “якорные” замены (пример)
    manual = {
        "Darth Vader": "Xarn Velgor",
        "Luke Skywalker": "Lio Skyrin",
        "Leia Organa": "Lea Orgen",
        "Han Solo": "Hann Solven",
        "Obi-Wan Kenobi": "Oben-Kan Kenari",
        "Yoda": "Yoro",
        "Palpatine": "Valpatin",
        "The Force": "Synth Flux",
        "Jedi": "Astra Monks",
        "Sith": "Umbra Order",
        "Lightsaber": "Photon Blade",
        "Death Star": "Void Core",
        "Galactic Empire": "Orion Dominion",
        "Rebel Alliance": "Free Systems Pact",
        "Tatooine": "Tessarune",
        "Coruscant": "Corvessant",
        "Millennium Falcon": "Millenial Kite",
    }
    mapping.update(manual)

    # Автогенерация для остальных страниц
    for t in pages:
        if t in mapping:
            continue
        # если похоже на имя персонажа — сгенерим что-то “именное”
        if re.search(r"\b[A-Z][a-z]+(\s[A-Z][a-z]+)+\b", t):
            mapping[t] = f"{fake.first_name()} {fake.last_name()}"
        else:
            # иначе делаем “инопланетный” слог + суффикс
            base = fake.word().capitalize()
            mapping[t] = f"{base}{fake.random_element(elements=['on','ar','ium','is','ea','os'])}"

    # Core terms (если не покрыты manual)
    for ct in CORE_TERMS:
        if ct not in mapping:
            mapping[ct] = f"{fake.word().capitalize()} {fake.word().capitalize()}"

    return mapping

def apply_replacements(text: str, mapping: Dict[str, str]) -> str:
    """
    Делаем замены аккуратно: сначала длинные ключи, потом короткие.
    Регистр частично сохраняем (простым способом).
    """
    # сортируем по длине ключа (чтобы "Star Wars" заменилось раньше "Wars")
    keys = sorted(mapping.keys(), key=len, reverse=True)

    def repl(match: re.Match, value: str) -> str:
        src = match.group(0)
        # простое сохранение капитализации
        if src.isupper():
            return value.upper()
        if src[0].isupper():
            return value[0].upper() + value[1:]
        return value.lower()

    out = text
    for k in keys:
        v = mapping[k]
        pattern = re.compile(re.escape(k), flags=re.IGNORECASE)
        out = pattern.sub(lambda m, vv=v: repl(m, vv), out)
    return out

def main():
    kb_dir = Path("knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)

    mapping = make_terms_map(PAGES)

    # сохраняем словарь замен
    Path("terms_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = 0
    for title in PAGES:
        raw = wiki_extract(title)
        if not raw:
            print(f"[WARN] empty extract: {title}")
            continue

        replaced = apply_replacements(raw, mapping)

        # простая проверка: чтобы не осталось самых узнаваемых токенов
        # (можно расширять список)
        forbidden = ["Darth", "Jedi", "Sith", "Skywalker", "Tatooine", "Wookiee", "Star Wars"]
        if any(f.lower() in replaced.lower() for f in forbidden):
            # не критично, но сигнал что надо усилить mapping
            print(f"[WARN] possible leak terms in: {title}")

        filename = slugify(mapping.get(title, title))[:80] + ".md"
        (kb_dir / filename).write_text(
            f"# {mapping.get(title, title)}\n\n" + replaced,
            encoding="utf-8"
        )
        ok += 1

    print(f"Done. Saved {ok} docs to {kb_dir}/ and terms_map.json")

if __name__ == "__main__":
    main()
