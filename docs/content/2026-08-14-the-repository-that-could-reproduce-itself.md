# The repository that could reproduce itself — but could not publish itself

**Whitepaper themes:** Rapid research integration — the flywheel; Building a high-performance MLX
inference engine in Swift

A public source release is easy to mistake for a directory copy.

Choose the files, leave out the private material, push the result, and call it reproducible. That
works until someone asks the question that matters: can the public repository, using only what it
contains, prove its own boundary and regenerate the same source distribution?

For fast-mlx, the answer used to be no. The engineering checkout could produce the public tree,
but the public tree did not contain enough identity information to reproduce itself. The export
was reviewed and deterministic, yet it was still a one-way operation.

We wanted the public repository to be independently useful to a contributor or commercial
evaluator without giving it permission to publish anything new. Those requirements sound similar.
They are deliberately different.

## The public checkout needed its own closed world

The engineering repository has material that does not belong in a public source distribution:
operator state, private evidence, acquisition controls, machine-local artifacts, and internal
research boundaries. Its development manifest therefore maps an explicit set of reviewed source
paths into public destinations.

The public checkout receives a different manifest. It names only identity paths—the paths that
actually exist in the public tree—and seals the complete sorted path-and-mode set. There is no
fallback to an ambient repository, no request for a private manifest, and no conditional bypass
for a familiar machine or organization.

The self-reproducing source release boundary contained 621 files. Of those, 619 used regular mode
`100644` and two retained owner-executable mode `100755`. That mode distinction matters. A copy
that silently turns a tool into a non-executable file is not the same source distribution, even if
every byte matches.

That release's public manifest sealed those 621 path/mode records. The exporter then reads both
metadata and content from Git's index rather than the mutable working tree. An untracked edit, a
different ambient checkout, or a file changed after staging cannot quietly enter the release.

## The identity had to avoid hashing itself

Putting a content hash of the manifest inside the manifest creates a recursion problem: changing
the hash changes the file, which changes the hash again.

fast-mlx separates two proofs instead.

The manifest carries a noncircular seal over the complete path/mode set. Exact content equality is
proved separately by the Git tree and by comparing the first export with a strict re-export from
the public checkout. Both generations must contain the same paths, modes, and bytes.

That split is less magical and more useful. The manifest answers, “What is allowed to exist?” The
Git tree answers, “What exact content was released?” The re-export answers, “Can the public source
reproduce that content without private input?”

For the self-reproducing release, all three answers agreed.

## Refusal had to happen before the first copy

The most important exporter behavior is not copying. It is refusing.

Before writing any output, the exporter validates the full manifest, every source and destination,
every expanded tree member, every Git mode, every published article source, and the complete public
index. It refuses private prefixes, nested Git metadata, symlinks, special files, duplicate
destinations, malformed paths, invalid UTF-8, unsupported modes, and tree expansions that become
private only after source and destination are joined.

That last case survived several earlier reviews. A tree destination could look safe in isolation
while one expanded member landed under a forbidden nested path. The final preflight checks the
joined destination of every member before copying anything.

Fresh negative controls exercised those boundaries. Each refused candidate left zero output files.
The positive candidate validated, passed the complete 194-test public suite, regenerated itself,
and matched the original projection exactly.

This is a useful property for any project that publishes a reviewed subset of a larger engineering
environment: a failure should produce no plausible partial release.

## Deployment received a separate proof

Source reproducibility does not prove that GitHub Pages served the same result. The publication
workflow therefore kept a separate same-commit chain.

The exact public commit passed the repository boundary, three dependency-free Swift targets, Pages
build, Pages deployment, and post-deploy publication verification—seven successful jobs in one
run attempt. The verifier rebuilt the static site and compared every reader route twice. The
release-entry deployment recorded 55 generated files, 54 reader routes, two complete matching
cohorts, and zero failures in an independently validated receipt.

That receipt proves the bytes observed during those cohorts. It does not promise permanent
availability, prove GitHub's internal state, or authorize a rollback.

## Reproduction is not self-publication

The repository can now reproduce its reviewed public boundary. It still cannot decide that a new
file, benchmark, article, model result, or runtime capability deserves publication.

Research notes enter the site only through an explicit reviewed manifest. Release records are
explicit too. The exporter cannot promote engineering evidence, discover new claims, ingest an
external result, or push a commit. Every authority flag in the publication receipt remains false.

That distinction is the real design result. A trustworthy improvement loop should automate
mechanical proof while keeping product judgment visible. Rebuilding, hashing, comparing, testing,
and refusing are mechanical. Deciding what the project claims is not.

The public repository became self-reproducing, but it did not become self-authorizing. That is what
makes the automation safe enough to keep improving.
