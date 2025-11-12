"""
Make kelly_pack executable as a module.
python -m kelly_pack [args...]
"""
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())


