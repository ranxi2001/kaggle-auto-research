"""Competition discussion and notebook scraper."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiscussionPost:
    title: str
    author: str
    votes: int
    url: str
    content_preview: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class NotebookInfo:
    title: str
    author: str
    score: float | None
    votes: int
    url: str
    kernel_ref: str = ""


class CompetitionScraper:
    """Scrape competition discussions and notebooks for insights."""

    def __init__(self, competition_slug: str):
        self.slug = competition_slug
        self.base_url = f"https://www.kaggle.com/competitions/{competition_slug}"

    def get_top_notebooks(self, limit: int = 10) -> list[NotebookInfo]:
        """Get top-scoring public notebooks."""
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "list",
                 "--competition", self.slug,
                 "--sort-by", "voteCount",
                 "--page-size", str(limit),
                 "--csv"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return []

            notebooks = []
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return []

            headers = lines[0].split(",")
            for line in lines[1:]:
                values = line.split(",")
                row = dict(zip(headers, values))
                notebooks.append(NotebookInfo(
                    title=row.get("title", ""),
                    author=row.get("author", ""),
                    score=self._parse_float(row.get("bestScore")),
                    votes=int(row.get("totalVotes", 0)),
                    url=f"https://www.kaggle.com/code/{row.get('ref', '')}",
                    kernel_ref=row.get("ref", ""),
                ))
            return notebooks
        except Exception:
            return []

    def get_competition_description(self) -> str:
        """Fetch competition overview/description via kaggle API."""
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "list", "-s", self.slug, "--csv"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    headers = lines[0].split(",")
                    values = lines[1].split(",")
                    row = dict(zip(headers, values))
                    return row.get("description", "")
        except Exception:
            pass
        return ""

    def generate_research_report(
        self,
        competition_info: dict,
        notebooks: list[NotebookInfo],
        output_path: Path,
    ) -> Path:
        """Generate a structured research report."""
        report_lines = [
            f"# Research Report: {competition_info.get('title', self.slug)}",
            "",
            "## Competition Summary",
            f"- **Slug**: {self.slug}",
            f"- **URL**: {self.base_url}",
            f"- **Category**: {competition_info.get('category', 'N/A')}",
            f"- **Metric**: {competition_info.get('evaluation_metric', 'N/A')}",
            f"- **Reward**: {competition_info.get('reward', 'N/A')}",
            f"- **Deadline**: {competition_info.get('deadline', 'N/A')}",
            f"- **Teams**: {competition_info.get('team_count', 'N/A')}",
            "",
            "## Data Files",
        ]

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "files", self.slug, "--csv"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    for line in lines[1:]:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            report_lines.append(f"- `{parts[0]}` ({parts[1]})")
                else:
                    report_lines.append("- No file info available")
            else:
                report_lines.append("- Unable to fetch file list")
        except Exception:
            report_lines.append("- Unable to fetch file list")

        report_lines.extend([
            "",
            "## Top Public Notebooks",
            "",
        ])

        if notebooks:
            for i, nb in enumerate(notebooks[:10], 1):
                score_str = f" | Score: {nb.score}" if nb.score else ""
                report_lines.append(
                    f"{i}. **{nb.title}** by {nb.author} "
                    f"(Votes: {nb.votes}{score_str})"
                )
                report_lines.append(f"   - {nb.url}")
        else:
            report_lines.append("- No public notebooks found")

        report_lines.extend([
            "",
            "## Recommended Strategy",
            "",
            "TODO: Fill in after analyzing top solutions",
            "",
            "## Key Insights",
            "",
            "TODO: Extract from discussion posts and notebooks",
            "",
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines))
        return output_path

    @staticmethod
    def _parse_float(val: str | None) -> float | None:
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None
