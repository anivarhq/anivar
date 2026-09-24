// Tauri refuses to build when an @tauri-apps npm package and its Rust crate
// are on different major/minor versions. Nothing in CI builds an installer, so
// that only ever surfaced at the moment of tagging a release: Dependabot bumped
// @tauri-apps/plugin-notification to 2.4.0 on the npm side alone, and v0.1.4's
// installers failed on all three platforms.
//
// This compares the two sides from the lockfiles, in a second.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

const npmLock = JSON.parse(readFileSync(join(root, 'package-lock.json'), 'utf8'))
const cargoLock = readFileSync(join(root, 'src-tauri', 'Cargo.lock'), 'utf8')

/** Resolved version of an npm package, or null when it isn't installed. */
function npmVersion(name) {
  const entry = npmLock.packages?.[`node_modules/${name}`]
  return entry?.version ?? null
}

/** Version of a crate as the Cargo lockfile resolved it. */
function crateVersion(name) {
  const match = cargoLock.match(
    new RegExp(`\\[\\[package\\]\\]\\nname = "${name}"\\nversion = "([^"]+)"`),
  )
  return match?.[1] ?? null
}

const minor = (version) => version.split('.').slice(0, 2).join('.')

// The npm package and the crate that have to move together.
const pairs = [
  ['@tauri-apps/api', 'tauri'],
  ['@tauri-apps/cli', 'tauri-cli'],
]
for (const name of Object.keys(npmLock.packages ?? {})) {
  const match = name.match(/^node_modules\/@tauri-apps\/plugin-(.+)$/)
  if (match) pairs.push([`@tauri-apps/plugin-${match[1]}`, `tauri-plugin-${match[1]}`])
}

const problems = []
for (const [npmName, crateName] of pairs) {
  const fromNpm = npmVersion(npmName)
  const fromCargo = crateVersion(crateName)
  // A pair only counts when both sides are actually present: the CLI, for
  // one, is an npm-only dependency here.
  if (!fromNpm || !fromCargo) continue
  if (minor(fromNpm) !== minor(fromCargo)) {
    problems.push(`  ${crateName} (v${fromCargo}) : ${npmName} (v${fromNpm})`)
  }
}

if (problems.length > 0) {
  console.error(
    'Tauri packages are on different major/minor versions, so `tauri build`\n' +
      'will refuse to run and the release installers will not be produced:\n' +
      problems.join('\n') +
      '\n\nBring the Rust side up with:\n' +
      '  cargo update -p <crate> --manifest-path src-tauri/Cargo.toml',
  )
  process.exit(1)
}

console.log(`Tauri npm packages and Rust crates agree (${pairs.length} pairs checked).`)
