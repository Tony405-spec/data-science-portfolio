# Dependency Update Policy

This project pins runtime, notebook, and development dependencies in `requirements.txt` for reproducible data-science runs.

## Update Principles

- Prefer small, reviewable dependency updates over broad untested upgrades.
- Keep model, preprocessing, Dask, and notebook dependencies pinned unless there is a clear compatibility reason to loosen them.
- Update pins only after running the relevant tests or documenting why a test could not be run locally.
- Treat major-version upgrades as behavior changes that need focused review.

## Review Checklist

Before opening a dependency update PR:

- Identify why the update is needed: bug fix, security fix, compatibility, or feature support.
- Read upstream release notes for breaking changes.
- Update `requirements.txt` and any affected documentation together.
- Run `python -m pytest` when the local environment has all required dependencies installed.
- For notebook-related packages, smoke-check notebook execution guidance if the update affects `jupyter`, `nbconvert`, `nbclient`, or kernels.
- For Dask-related updates, run or document the distributed pipeline test result.
- Include known migration notes in the PR description.

## Security Updates

Security updates should be prioritized, but they should still be tested. If a vulnerable dependency cannot be upgraded safely in one step, open a tracking issue that explains:

- The affected package and vulnerability class.
- The blocking compatibility issue.
- The planned mitigation or staged upgrade path.

Do not hide failing tests or remove security checks to force an upgrade through.
