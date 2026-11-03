"""Release check: every report must rebuild from released Parquet in under 10 minutes on the M1.

    python scripts/release_check.py <released-parquet-root> <output-dir>
"""

import sys

from whopays.report.analysis import timed_regenerate


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    result = timed_regenerate(sys.argv[1], sys.argv[2])
    print(result)
    return 0 if result["within_limit"] else 1


if __name__ == "__main__":
    sys.exit(main())
