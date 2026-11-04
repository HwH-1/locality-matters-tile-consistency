"""Single source of truth for paths inside this release."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
TC = CODE / "tc"
REFPARENT = CODE / "reference"
REF = REFPARENT / "src"
RECORDS = ROOT / "data" / "records"
IMAGES = ROOT / "data" / "images"
BENCHMARK = ROOT / "data" / "benchmark"
FIGDIR = ROOT / "figures"
MANUSCRIPT = ROOT / "manuscript"
