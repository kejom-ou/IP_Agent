"""
本地模型状态检查
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_models.model_downloader import check_all

if __name__ == "__main__":
    check_all()
