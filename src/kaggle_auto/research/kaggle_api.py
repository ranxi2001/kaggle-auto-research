"""Kaggle API wrapper for competition research."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompetitionInfo:
    slug: str
    title: str
    category: str
    reward: str
    deadline: str
    team_count: int
    evaluation_metric: str
    description: str = ""
    url: str = ""


@dataclass
class DatasetInfo:
    name: str
    size: str
    file_count: int
    files: list[dict]


class KaggleAPI:
    """Wrapper around kaggle CLI for competition interaction."""

    def __init__(self):
        import sys
        self._cmd_prefix = [sys.executable, "-m", "kaggle"]
        self._verify_credentials()

    def _verify_credentials(self):
        """Check if kaggle CLI is configured."""
        import os
        if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_USERNAME"):
            return
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            return
        result = subprocess.run(
            self._cmd_prefix + ["config", "view"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Kaggle CLI not configured. Set KAGGLE_API_TOKEN env var or create ~/.kaggle/kaggle.json"
            )

    def _run(self, args: list[str]) -> str:
        result = subprocess.run(
            self._cmd_prefix + args,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"kaggle {' '.join(args)} failed: {result.stderr}")
        return result.stdout

    def list_competitions(self, search: str = "", category: str = "") -> list[dict]:
        """List active competitions."""
        args = ["competitions", "list", "--csv"]
        if search:
            args += ["-s", search]
        if category:
            args += ["--category", category]

        output = self._run(args)
        return self._parse_csv(output)

    def get_competition_info(self, slug: str) -> CompetitionInfo:
        """Get detailed competition information."""
        output = self._run(["competitions", "list", "--csv", "-s", slug])
        rows = self._parse_csv(output)

        for row in rows:
            if row.get("ref", "") == slug or slug in row.get("ref", ""):
                return CompetitionInfo(
                    slug=row.get("ref", slug),
                    title=row.get("title", ""),
                    category=row.get("category", ""),
                    reward=row.get("reward", ""),
                    deadline=row.get("deadline", ""),
                    team_count=int(row.get("teamCount", 0)),
                    evaluation_metric=row.get("evaluationMetric", ""),
                    url=f"https://www.kaggle.com/competitions/{row.get('ref', slug)}",
                )

        return CompetitionInfo(
            slug=slug,
            title=slug,
            category="",
            reward="",
            deadline="",
            team_count=0,
            evaluation_metric="",
            url=f"https://www.kaggle.com/competitions/{slug}",
        )

    def list_files(self, competition_slug: str) -> list[dict]:
        """List competition data files."""
        output = self._run(["competitions", "files", competition_slug, "--csv"])
        return self._parse_csv(output)

    def download_data(self, competition_slug: str, dest: Path) -> None:
        """Download competition data."""
        dest.mkdir(parents=True, exist_ok=True)
        self._run([
            "competitions", "download", competition_slug,
            "-p", str(dest),
        ])

    def list_kernels(self, competition_slug: str, sort_by: str = "scoreAscending") -> list[dict]:
        """List public notebooks/kernels for a competition."""
        output = self._run([
            "kernels", "list",
            "--competition", competition_slug,
            "--sort-by", sort_by,
            "--csv",
        ])
        return self._parse_csv(output)

    def submit(self, competition_slug: str, file_path: Path, message: str = "") -> str:
        """Submit predictions to a competition."""
        args = [
            "competitions", "submit",
            competition_slug,
            "-f", str(file_path),
            "-m", message or "Auto submission via kaggle-auto-research",
        ]
        return self._run(args)

    def get_submissions(self, competition_slug: str) -> list[dict]:
        """Get submission history."""
        output = self._run(["competitions", "submissions", competition_slug, "--csv"])
        return self._parse_csv(output)

    def pull_kernel(self, kernel_ref: str, dest: Path) -> None:
        """Pull a public notebook with metadata."""
        dest.mkdir(parents=True, exist_ok=True)
        self._run(["kernels", "pull", kernel_ref, "-p", str(dest), "-m"])

    def _parse_csv(self, csv_text: str) -> list[dict]:
        """Parse CSV output from kaggle CLI."""
        lines = csv_text.strip().split("\n")
        if len(lines) < 2:
            return []
        headers = lines[0].split(",")
        rows = []
        for line in lines[1:]:
            values = line.split(",")
            row = dict(zip(headers, values))
            rows.append(row)
        return rows
