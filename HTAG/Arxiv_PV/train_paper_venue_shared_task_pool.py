import os
import sys

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from train_paper_venue_shared import main  # noqa: E402


if __name__ == "__main__":
    main()
