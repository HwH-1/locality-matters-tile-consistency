"""Re-run only the queries that hit OpenAlex rate limiting (429), with backoff.

Merges results back into state/m2_search_log.json in place, keyed by query id.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"
MAILTO = "research@example.org"
LOG = "/data/tileconsistency/state/m2_search_log.json"
RETRY = {"q10", "q17", "q18", "q19", "q20"}


def fetch(flt, per_page=8, attempts=4):
    url = (f"{BASE}?filter={urllib.parse.quote(flt)}&per-page={per_page}"
           f"&sort=cited_by_count:desc&mailto={MAILTO}")
    delay = 8.0
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"sci-m2-probe ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                d = json.load(fh)
            return {"filter": flt, "count": d["meta"]["count"],
                    "top": [{"title": w["display_name"], "year": w["publication_year"],
                             "cited": w["cited_by_count"],
                             "doi": (w.get("doi") or "").replace("https://doi.org/", "")}
                            for w in d["results"]]}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            return {"filter": flt, "error": f"HTTP {e.code}"}
        except Exception as e:                      # noqa: BLE001
            return {"filter": flt, "error": str(e)}
    return {"filter": flt, "error": "HTTP 429 after retries"}


def main():
    log = json.load(open(LOG))
    for q in log["queries"]:
        if q["id"] not in RETRY:
            continue
        res = fetch(q["filter"])
        q.pop("error", None)
        q.update(res)
        print(f'{q["id"]}: count={res.get("count", res.get("error")):>6}  '
              f'{q["question"][:66]}', flush=True)
        time.sleep(12.0)
        with open(LOG, "w") as fh:
            json.dump(log, fh, indent=2, ensure_ascii=False)
    log["retried_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG, "w") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print("merged into", LOG)


if __name__ == "__main__":
    main()
