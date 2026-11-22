"""PatchCore용 기존 명령을 유지하는 공통 임계값 평가기의 호환 진입점."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.evaluate_threshold import main

if __name__ == "__main__":
    main()
