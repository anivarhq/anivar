# Code signing policy

Anivar's Windows installers are not code-signed yet. This page describes how
they will be signed, and who decides, once signing is in place.

## What gets signed

Only the release artifacts the public
[release workflow](.github/workflows/release.yml) builds from a `v*` tag on
`main`:

- `Anivar_<version>_x64-setup.exe` (the NSIS installer) and the `anivar.exe`
  inside it.

Nothing built on a developer's machine is ever signed. Every signed file can be
traced to its tag, its commit and the GitHub Actions run that built it, and
every release publishes a SHA-256 checksum beside each file.

## Team roles

| Role | Who | What they do |
|---|---|---|
| Committer and reviewer | [@RANJITH1708](https://github.com/RANJITH1708) | Merges reviewed changes to `main`. Every change, including from outside contributors, goes through a pull request and must pass CI on Windows, macOS and Linux. |
| Approver | [@RANJITH1708](https://github.com/RANJITH1708) | Approves each signing request by hand, after checking it came from a release tag's workflow run. |

Everyone with one of these roles uses multi-factor authentication on GitHub and
on the signing service.

## Updates

Separately from installer signing, every in-app update is signed with a
minisign key and checked by the app against the public key built into it before
anything is installed (see `src-tauri/src/update_cmds.rs`).

## Privacy

What Anivar sends, and when, is in [PRIVACY.md](PRIVACY.md). In short: nothing
leaves your computer unless you turn on a feature that sends it.

## Reporting a problem

Something signed that shouldn't be, or a signature that doesn't verify: please
report it privately through [GitHub's advisory form](https://github.com/anivarhq/anivar/security/advisories/new).
