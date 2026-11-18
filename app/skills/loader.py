"""Load project SKILL.md files into StateBackend-compatible file data."""

from typing import Any

from deepagents.backends.utils import create_file_data

from app.config import BASE_DIR


def load_skill_files() -> dict[str, Any]:
    """Load every skills/*/SKILL.md file into virtual backend paths."""

    skills_dir = BASE_DIR / "skills"
    skill_files: dict[str, Any] = {}

    if not skills_dir.exists():
        raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        virtual_path = "/" + skill_path.relative_to(BASE_DIR).as_posix()
        content = skill_path.read_text(encoding="utf-8")
        skill_files[virtual_path] = create_file_data(content)

    if not skill_files:
        raise FileNotFoundError(
            f"No SKILL.md files found inside {skills_dir}."
        )

    return skill_files
