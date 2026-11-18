from pathlib import Path

import pytest

from app.skills import loader


def test_loads_all_skill_files(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"

    expected_skills = {
        "research": "# Research Skill",
        "software-development": "# Software Development Skill",
        "code-review": "# Code Review Skill",
    }

    for name, content in expected_skills.items():
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    monkeypatch.setattr(loader, "BASE_DIR", tmp_path)

    result = loader.load_skill_files()

    expected_paths = {
        f"/skills/{name}/SKILL.md"
        for name in expected_skills
    }

    assert set(result.keys()) == expected_paths
    assert all(value is not None for value in result.values())


def test_skill_content_is_loaded_correctly(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills" / "research"
    skills_dir.mkdir(parents=True)

    skill_content = "# Research Skill\nResearch technical topics."

    (skills_dir / "SKILL.md").write_text(
        skill_content,
        encoding="utf-8",
    )

    monkeypatch.setattr(loader, "BASE_DIR", tmp_path)

    # Replace the Deep Agents converter to inspect the input content.
    monkeypatch.setattr(
        loader,
        "create_file_data",
        lambda content: content,
    )

    result = loader.load_skill_files()

    assert result["/skills/research/SKILL.md"] == skill_content


def test_missing_skills_directory_raises_error(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "BASE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_skill_files()


def test_empty_skills_directory_raises_error(tmp_path, monkeypatch):
    (tmp_path / "skills").mkdir()

    monkeypatch.setattr(loader, "BASE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_skill_files()