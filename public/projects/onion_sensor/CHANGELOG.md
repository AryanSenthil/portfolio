# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with Python 3.10.12
- Virtual environment configuration (.venv)
- UV package manager integration
- Automated dependency management
- Git repository initialization
- Code quality tools (black, ruff, mypy)
- Testing framework (pytest)
- Multiple AI assistant setup scripts (setup_claude.sh, setup_cursor.sh, setup_codex.sh)
- Jupyter notebook support for Codex AI
- Mandatory project structure enforcement (src/, tests/, docs/, draft/)
- Project structure creation in all setup scripts
- Comprehensive test suite for tools.py module (28 tests, all passing)
- Simplified test suite for classification_module.py module (14 tests, all passing)
- Test coverage for all major functions in both modules
- Error handling tests for missing files and invalid inputs
- Created model_agent module in src/ directory
- Added __init__.py file for model_agent module
- Added precise inline comments to classification_module.py functions with purpose, args, outputs, and requirements
- Added precise inline comments to regression_module.py functions with purpose, args, outputs, data format, metrics, and requirements
- Added letta-client package (v0.1.324) to requirements.txt for agent orchestration

### Changed
- Renamed setup.sh to setup_claude.sh for clarity
- Renamed setup-cursor.sh to setup_cursor.sh for consistency
- Updated setup scripts with appropriate AI assistant configurations
- Enforced mandatory project structure in .cursorrules
- Updated all setup scripts to create proper project structure (src/, tests/, docs/, draft/)
- Added project structure enforcement rules for all AI assistants

### Deprecated

### Removed

### Fixed

### Security

---

## [0.1.0] - YYYY-MM-DD

### Added
- Project initialization

---

## Template for Future Entries

Copy this template for new versions:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security-related changes
```

---

## Guidelines

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes or security improvements

Keep entries:
- Brief but descriptive
- User-focused (what changed for them)
- Grouped by type
- Chronologically ordered (newest first)

When adding dependencies, format as:
```
### Added
- Dependency: package-name (v1.2.3) - Brief reason for addition
```

When updating dependencies, format as:
```
### Changed
- Updated package-name from v1.2.3 to v1.2.4 - Brief reason for update
```