"""M2 prior-art probe: reproducible OpenAlex searches that bound the novelty claim.

Syntax note: OpenAlex search filters take ONE value; boolean operators live
INSIDE the value ("a AND b", "a OR b", quoted phrases).  Repeating the filter
key inside the value searches for the literal key text and returns 0 hits --
that mistake is why this file states the grammar explicitly.

Every query is stored with its timestamp, the exact filter string, the total hit
count and the top by-citation titles, so that the "is anyone else working on
this?" answer in the paper is backed by a re-runnable log rather than by
recollection.

Output: state/m2_search_log.json
"""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"
MAILTO = "research@example.org"

# (query_id, the question the query is meant to answer, filter value)
QUERIES = [
    ("q01", "is tile/part consistency named as a property of an image filter?",
     'title_and_abstract.search:"tile consistency"'),
    ("q02", "tile + overlap + seam in a WSI/virtual-staining context (closest lane)",
     'title_and_abstract.search:"tile overlap"'),
    ("q03", "seams studied in stylization/abstraction specifically?",
     'title_and_abstract.search:seam AND stylization'),
    ("q04", "gigapixel + abstraction / style transfer",
     'title_and_abstract.search:gigapixel AND stylization'),
    ("q05", "gigapixel + non-photorealistic rendering",
     'title_and_abstract.search:gigapixel AND "non-photorealistic"'),
    ("q06", "tiling error attributed to global statistics (vs receptive field)?",
     'title_and_abstract.search:tiling AND "global statistics"'),
    ("q07", "overlap-tile strategy: how large is this literature?",
     'title_and_abstract.search:"overlap tile"'),
    ("q08", "halo / margin terminology in tiled processing",
     'title_and_abstract.search:"tile halo" OR "halo size"'),
    ("q09", "Kuwahara + adaptive: the direct competitor lane",
     'title_and_abstract.search:Kuwahara AND adaptive'),
    ("q10", "adaptive window/size selection for Kuwahara specifically",
     'title.search:Kuwahara'),
    ("q11", "percentile/quantile normalisation inside a local filter",
     'title_and_abstract.search:percentile AND "local statistics"'),
    ("q12", "structure tensor normalisation by an image-wide quantity",
     'title_and_abstract.search:"structure tensor" AND normalization'),
    ("q13", "streaming / out-of-core filtering with correctness guarantees",
     'title_and_abstract.search:streaming AND "out-of-core"'),
    ("q14", "reproducibility of style-transfer / NPR evaluation protocols",
     'title_and_abstract.search:reproducibility AND "style transfer" AND evaluation'),
    ("q15", "evaluation protocol changing method ranking in IQA",
     'title_and_abstract.search:ranking AND "image quality assessment" AND protocol'),
    ("q16", "whole-slide preprocessing with stain/statistics normalisation + tiling",
     'title_and_abstract.search:"whole slide" AND normalization AND tiling'),
    ("q17", "control: volume of the general tiled-large-image lane",
     'title_and_abstract.search:tiling AND "large images" AND artifacts'),
    ("q18", "translation variance / equivariance as the stated cause of tiling error",
     'title_and_abstract.search:"translation variance"'),
    ("q19", "seam removal / blending as the standard remedy (what we argue is insufficient)",
     'title_and_abstract.search:seam AND blending AND tiles'),
    ("q20", "content-adaptive filters whose parameter depends on image-wide statistics",
     'title_and_abstract.search:"content adaptive" AND statistics'),
]


def fetch(flt, per_page=8):
    url = (f"{BASE}?filter={urllib.parse.quote(flt)}&per-page={per_page}"
           f"&sort=cited_by_count:desc&mailto={MAILTO}")
    req = urllib.request.Request(url, headers={"User-Agent": f"sci-m2-probe ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        d = json.load(fh)
    return {
        "filter": flt,
        "count": d["meta"]["count"],
        "top": [{"title": w["display_name"],
                 "year": w["publication_year"],
                 "cited": w["cited_by_count"],
                 "doi": (w.get("doi") or "").replace("https://doi.org/", "")}
                for w in d["results"]],
    }


def main():
    log = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": "OpenAlex works API (api.openalex.org/works)",
           "syntax": "one filter value; booleans inside the value",
           "queries": []}
    for qid, question, flt in QUERIES:
        try:
            res = fetch(flt)
        except Exception as e:                      # noqa: BLE001
            res = {"filter": flt, "error": str(e)}
        log["queries"].append({"id": qid, "question": question, **res})
        print(f"{qid}: count={res.get('count', 'ERR'):>6}  {question[:72]}", flush=True)
        time.sleep(0.4)
    out = "/data/tileconsistency/state/m2_search_log.json"
    with open(out, "w") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
