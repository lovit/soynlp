"""Install wheel and sdist in isolated Python 3.12 environments and check resources."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    distributions = [*sorted((ROOT / "dist").glob("*.whl")), *sorted((ROOT / "dist").glob("*.tar.gz"))]
    if len(distributions) != 2:
        raise ValueError("Run uv build first; dist must contain one wheel and one sdist")
    with tempfile.TemporaryDirectory(prefix="soynlp-install-") as temporary:
        workspace = Path(temporary)
        requirements = workspace / "requirements.txt"
        requirements.write_text(
            run(
                [
                    uv,
                    "export",
                    "--locked",
                    "--no-dev",
                    "--extra",
                    "emoji",
                    "--no-emit-project",
                    "--format",
                    "requirements-txt",
                ],
                ROOT,
            )
        )
        for index, distribution in enumerate(distributions):
            environment = workspace / f"env-{index}"
            run([uv, "venv", "--python", "3.12", str(environment)], workspace)
            python = environment / "bin" / "python"
            run([uv, "pip", "install", "--python", str(python), "--requirements", str(requirements)], workspace)
            run([uv, "pip", "install", "--python", str(python), "--no-deps", str(distribution)], workspace)
            check = """import sys
from pathlib import Path
import soynlp
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor
from soynlp.postagger import KoreanPOSTagger, LRMaxScoreTagger
from soynlp.services import text_processor
assert Path(soynlp.__file__).is_relative_to(Path(sys.prefix))
assert LRNounExtractor(verbose=False).extract(["한국어를 공부합니다", "한국어는 재미있습니다"])
LRNounExtractor_v2(verbose=False)
NewsNounExtractor(verbose=False)
assert KoreanPOSTagger().tag("한국어를 공부합니다")
assert LRMaxScoreTagger().pos("한국어를 공부합니다")
assert ("먹", "었다") in text_processor("lemmatize")("먹었다")
assert text_processor("normalize", "emoji")("한국어 😀") == "한국어 "
"""
            run([str(python), "-c", check], workspace)
            for command in ([str(python), "-m", "soynlp"], [str(environment / "bin" / "soynlp")]):
                result = json.loads(run([*command, "normalize", "--method", "repeat", "--text", "한국어ㅋㅋㅋㅋ"], workspace))
                assert result["normalized"] == "한국어ㅋㅋ"
            print(f"Installed and verified {distribution.name}")


if __name__ == "__main__":
    main()
