"""Pre-submission validation."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, is_zipfile

import numpy as np
import pandas as pd
import yaml


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class SubmissionValidator:
    """Validate submission format before uploading."""

    def validate(
        self,
        submission_path: Path,
        sample_submission_path: Path,
    ) -> ValidationResult:
        """Validate submission against sample_submission format."""
        errors = []
        warnings = []

        if not submission_path.exists():
            return ValidationResult(False, ["Submission file does not exist"], [])

        if not sample_submission_path.exists():
            return ValidationResult(False, ["Sample submission file does not exist"], [])

        try:
            sub = pd.read_csv(submission_path)
            sample = pd.read_csv(sample_submission_path)
        except Exception as e:
            return ValidationResult(False, [f"Failed to read CSV: {e}"], [])

        if list(sub.columns) != list(sample.columns):
            errors.append(
                f"Column mismatch. Expected: {list(sample.columns)}, "
                f"Got: {list(sub.columns)}"
            )

        if len(sub) != len(sample):
            errors.append(
                f"Row count mismatch. Expected: {len(sample)}, Got: {len(sub)}"
            )

        if sub.isnull().any().any():
            null_cols = sub.columns[sub.isnull().any()].tolist()
            errors.append(f"Null values found in columns: {null_cols}")

        id_col = sample.columns[0]
        if id_col in sub.columns and id_col in sample.columns:
            if not sub[id_col].equals(sample[id_col]):
                if set(sub[id_col]) != set(sample[id_col]):
                    errors.append(f"ID column '{id_col}' values don't match sample")
                else:
                    warnings.append(f"ID column '{id_col}' order differs from sample")

        pred_cols = sample.columns[1:]
        for col in pred_cols:
            if col in sub.columns:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    if sub[col].min() < -1e10 or sub[col].max() > 1e10:
                        warnings.append(f"Column '{col}' has extreme values")
                    if np.isinf(sub[col]).any():
                        errors.append(f"Column '{col}' contains infinity")

        if sub.duplicated(subset=[id_col]).any() if id_col in sub.columns else False:
            errors.append("Duplicate IDs found")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_skill_zip(self, submission_path: Path) -> ValidationResult:
        """Validate a static-skill ZIP without extracting it."""
        if not submission_path.exists():
            return ValidationResult(False, ["Submission file does not exist"], [])
        if not is_zipfile(submission_path):
            return ValidationResult(False, ["Submission is not a valid ZIP archive"], [])

        errors: list[str] = []
        warnings: list[str] = []
        try:
            with ZipFile(submission_path) as archive:
                infos = archive.infolist()
                if not infos:
                    errors.append("Submission ZIP is empty")

                names: set[str] = set()
                skill_manifests: list[tuple] = []
                total_size = sum(info.file_size for info in infos)
                encrypted = False
                for info in infos:
                    name = info.filename
                    if name in names:
                        errors.append(f"Duplicate ZIP member: {name}")
                    names.add(name)
                    if "\\" in name:
                        errors.append(f"ZIP member uses a backslash path: {name}")
                        continue
                    path = PurePosixPath(name)
                    if path.is_absolute() or ".." in path.parts:
                        errors.append(f"Unsafe ZIP member path: {name}")
                        continue
                    if path.parts and path.parts[0] != "skills":
                        errors.append(f"ZIP member must be rooted under skills/: {name}")
                    if info.flag_bits & 0x1:
                        encrypted = True
                        errors.append(f"Encrypted ZIP member is not allowed: {name}")
                    if len(path.parts) == 3 and path.parts[0] == "skills" and path.name == "SKILL.md":
                        skill_manifests.append((info, path))

                if total_size > 100 * 1024 * 1024:
                    errors.append("Uncompressed ZIP content exceeds 100 MiB")
                elif not encrypted:
                    corrupt = archive.testzip()
                    if corrupt:
                        errors.append(f"Corrupt ZIP member: {corrupt}")
                if not skill_manifests:
                    errors.append("ZIP must contain at least one skills/<name>/SKILL.md")
                for info, path in skill_manifests:
                    if info.file_size == 0:
                        errors.append(f"Skill manifest is empty: {info.filename}")
                    else:
                        try:
                            manifest = archive.read(info).decode("utf-8")
                        except UnicodeDecodeError:
                            errors.append(f"Skill manifest is not UTF-8: {info.filename}")
                        else:
                            errors.extend(self._validate_skill_manifest(manifest, path))
        except (BadZipFile, OSError, RuntimeError) as exc:
            errors.append(f"Failed to inspect ZIP: {exc}")

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _validate_skill_manifest(manifest: str, path: PurePosixPath) -> list[str]:
        """Validate required SKILL.md YAML frontmatter."""
        errors: list[str] = []
        lines = manifest.splitlines()
        if not lines or lines[0].strip() != "---":
            return [f"Skill manifest lacks YAML frontmatter: {path}"]

        try:
            closing_index = next(
                index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
            )
        except StopIteration:
            return [f"Skill manifest has unterminated YAML frontmatter: {path}"]

        try:
            metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
        except yaml.YAMLError as exc:
            return [f"Skill manifest has invalid YAML frontmatter ({path}): {exc}"]

        if not isinstance(metadata, dict):
            return [f"Skill manifest frontmatter must be a mapping: {path}"]

        expected_name = path.parts[1]
        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"Skill manifest is missing a non-empty name: {path}")
        elif name != expected_name:
            errors.append(
                f"Skill manifest name '{name}' does not match directory '{expected_name}': {path}"
            )
        if not isinstance(description, str) or not description.strip():
            errors.append(f"Skill manifest is missing a non-empty description: {path}")
        return errors
