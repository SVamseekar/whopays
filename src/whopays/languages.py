"""The 11 audit languages (PRD §Scope): those present in both MGSM and Belebele."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # MGSM code; the canonical key everywhere in this package
    flores: str  # Belebele / FLORES-200 code
    name: str
    script: str


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        Language("en", "eng_Latn", "English", "Latin"),
        Language("de", "deu_Latn", "German", "Latin"),
        Language("fr", "fra_Latn", "French", "Latin"),
        Language("es", "spa_Latn", "Spanish", "Latin"),
        Language("ru", "rus_Cyrl", "Russian", "Cyrillic"),
        Language("zh", "zho_Hans", "Chinese (Simplified)", "Han"),
        Language("ja", "jpn_Jpan", "Japanese", "Kana/Kanji"),
        Language("th", "tha_Thai", "Thai", "Thai"),
        Language("bn", "ben_Beng", "Bengali", "Bengali"),
        Language("sw", "swh_Latn", "Swahili", "Latin"),
        Language("te", "tel_Telu", "Telugu", "Telugu"),
    )
}

CODES: tuple[str, ...] = tuple(LANGUAGES)
