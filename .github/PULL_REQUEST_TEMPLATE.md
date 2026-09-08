<!--
Thanks for the PR! Please fill in the sections below. Anything not applicable
can be deleted.
-->

## Summary

<!-- One or two sentences describing what this PR changes and why. -->

## Related issue

<!-- e.g. Closes #123, Refs #456. Delete if not applicable. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Documentation
- [ ] Build / CI / tooling

## Test plan

<!--
How did you verify this works? Manual repro steps, screenshots, logs from
`cargo check` / `npm run build`, etc.
-->

## Checklist

- [ ] `cargo check` in `src-tauri/` is green
- [ ] `cargo clippy --lib --no-deps -- -D warnings` is green
- [ ] `npm run build` succeeds
- [ ] No secrets / personal data in the diff
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (if user-visible)
- [ ] `docs/ARCHITECTURE.md` updated (if a Rust subsystem changed)
