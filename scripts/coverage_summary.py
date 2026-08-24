#!/usr/bin/env python3
"""Print a JaCoCo coverage summary from the generated XML report.

Reads build/reports/jacoco/test/jacocoTestReport.xml (or a path given as the
first argument) and prints one line per counter type with covered/total and the
percentage. Exits non-zero if the report is missing so `make coverage` fails
loudly rather than silently printing nothing.
"""
import sys
import xml.etree.ElementTree as ET

DEFAULT_REPORT = "build/reports/jacoco/test/jacocoTestReport.xml"
ORDER = ["INSTRUCTION", "BRANCH", "LINE", "COMPLEXITY", "METHOD", "CLASS"]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPORT
    try:
        root = ET.parse(path).getroot()
    except (FileNotFoundError, ET.ParseError) as exc:
        print(f"coverage report unavailable ({path}): {exc}", file=sys.stderr)
        return 1

    counters = {c.get("type"): c for c in root.findall("counter")}
    print("Coverage summary:")
    for typ in ORDER:
        c = counters.get(typ)
        if c is None:
            continue
        missed = int(c.get("missed"))
        covered = int(c.get("covered"))
        total = missed + covered
        pct = 100.0 * covered / total if total else 0.0
        print(f"  {typ.lower():12} {covered:5}/{total:<5} {pct:6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
