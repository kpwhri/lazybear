# Changelog

Notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Better error message when `pl.col` rather than `col` is used

### Fixed

- Cleaned up `join`, dropping suffixes and adding prefix/suffix as well as options for how to handle duplicates (
  `rename` vs `drop`).
- Issue where sequence parameter would treat string as a list

## [0.3.0] - 2026-05-06

### Changed

- Altered license documentation in `pyproject.toml` to use SPDX license expression
- Added a database-specific result cleaning dispatch, currently used to strip trailing whitespace from teradata/
  `teradatasql` string/utf8 results

## [0.2.1] - 2026-04-07

### Added

- Documentation for github pages in `docs` directory, including API Reference and Expressions.
- Documentation link for package on PyPI

## [0.2.0] - 2026-01-26

### Added

- More tests around sorting
- Ability to upload temp tables for joining (beta, limited db support)

### Changed

- Fixed how arguments are handled when sorting by multiple keys

### Fixed

- Apply tablename changes (e.g., for SQL Server temp table) in the init of TempLazyBearFrame
- Ensuring that teradata has `ON COMMIT PRESERVE ROWS` appended during temp table creation

## [0.1.1] - 2026-01-13

### Added

- README.md with documentation and examples
- Workflows for publishing to pypi

## [0.1.0] - 2026-01-12

### Changed

- Separated out as a distinct repo

[unreleased]: https://github.com/kpwhri/lazybear/compare/v0.3.0..HEAD

[0.3.0]: https://github.com/kpwhri/lazybear/compare/v0.2.1..v0.3.0

[0.2.1]: https://github.com/kpwhri/lazybear/compare/v0.2.0..v0.2.1

[0.2.0]: https://github.com/kpwhri/lazybear/compare/v0.1.1..v0.2.0

[0.1.1]: https://github.com/kpwhri/lazybear/compare/v0.1.0..v0.1.1

[0.1.0]: https://github.com/kpwhri/lazybear/releases/tag/v0.1.0
