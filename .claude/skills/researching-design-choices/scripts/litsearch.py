"""Scholarly search through the OpenAlex API (free, no key): search, snowball, verify.

    python litsearch.py search "mixed precision quantization multilingual" --since 2023 -n 25
    python litsearch.py refs  <id-or-doi-or-arxiv>   # backward snowball: what it cites
    python litsearch.py cites <id-or-doi-or-arxiv>   # forward snowball: who cites it
    python litsearch.py check <id-or-doi-or-arxiv> ...  # confirm references exist

Prints one row per work: OpenAlex id, year, citations, venue (or "preprint"), title, DOI or
arXiv link. Use it to build the candidate list cheaply before opening anything in full.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

API = "https://api.openalex.org"
FIELDS = "id,display_name,publication_year,cited_by_count,doi,primary_location,ids,type"


def _get(path: str, **params) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "litsearch/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _resolve(ref: str) -> str:
    ref = ref.strip()
    arxiv = re.search(r"(\d{4}\.\d{4,5})", ref)
    if ref.startswith("W") or "openalex.org/W" in ref:
        return ref.rsplit("/", 1)[-1]
    if ref.startswith("10.") or "doi.org" in ref:
        return "doi:" + ref.split("doi.org/")[-1]
    if arxiv:
        return "doi:10.48550/arXiv." + arxiv.group(1)
    raise SystemExit(f"cannot resolve {ref!r}; give an OpenAlex W id, DOI or arXiv id")


def _row(w: dict) -> str:
    loc = (w.get("primary_location") or {}).get("source") or {}
    venue = loc.get("display_name") or "unknown venue"
    if "arxiv" in venue.lower() or w.get("type") == "preprint":
        venue = "preprint (arXiv)" if "arxiv" in venue.lower() else "preprint"
    link = w.get("doi") or ""
    return (f"{w['id'].rsplit('/', 1)[-1]}\t{w.get('publication_year')}\t"
            f"{w.get('cited_by_count', 0)}\t{venue[:40]}\t{w['display_name'][:110]}\t{link}")


def _print(works: list[dict]) -> None:
    print("id\tyear\tcites\tvenue\ttitle\tlink")
    for w in works:
        print(_row(w))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--since", type=int)
    s.add_argument("-n", type=int, default=25)
    s.add_argument("--sort", default="relevance_score:desc",
                   help="or cited_by_count:desc, publication_date:desc")
    for name in ("refs", "cites"):
        q = sub.add_parser(name)
        q.add_argument("ref")
        q.add_argument("-n", type=int, default=25)
    c = sub.add_parser("check")
    c.add_argument("refs", nargs="+")
    a = p.parse_args()

    if a.cmd == "search":
        flt = f"from_publication_date:{a.since}-01-01" if a.since else None
        params = {"search": a.query, "per-page": a.n, "select": FIELDS, "sort": a.sort}
        if flt:
            params["filter"] = flt
        _print(_get("works", **params)["results"])
    elif a.cmd == "refs":
        work = _get(f"works/{_resolve(a.ref)}", select="referenced_works")
        ids = [r.rsplit("/", 1)[-1] for r in work.get("referenced_works", [])][: a.n]
        if ids:
            _print(_get("works", filter="openalex:" + "|".join(ids), select=FIELDS,
                        **{"per-page": len(ids)})["results"])
    elif a.cmd == "cites":
        wid = _get(f"works/{_resolve(a.ref)}", select="id")["id"].rsplit("/", 1)[-1]
        _print(_get("works", filter=f"cites:{wid}", select=FIELDS, sort="cited_by_count:desc",
                    **{"per-page": a.n})["results"])
    else:
        missing = 0
        for ref in a.refs:
            try:
                print("FOUND\t" + _row(_get(f"works/{_resolve(ref)}", select=FIELDS)))
            except Exception as exc:  # noqa: BLE001 - report every failure, keep going
                missing += 1
                print(f"MISSING\t{ref}\t{exc}")
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
