# Changelog

All notable changes to the SEU Injection Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.8] - 2025-11-11

- **Branching Rework**: `dev` branch created and most older branches removed
- **Markdown Cleanup**: development markdown files kept on `dev` branch but removed from `main`
- **versions_and_plan.md created**: manual tracking of ideas going forward

## [1.1.7] - 2025-11-11

### Fixed
- **Release Workflow**: Optimized wheel verification to use CPU-only PyTorch
- **Disk Space**: Added cleanup steps to prevent out-of-space errors during builds
- **Version Consistency**: Ensured all version numbers are synchronized across files

### Changed
- **CI/CD**: Improved release workflow with best practices from popular PyTorch packages
- **Verification**: Streamlined wheel verification process for faster builds

*Please refer to `dev` branch for earlier changes*