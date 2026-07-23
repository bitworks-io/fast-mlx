#!/bin/zsh
set -euo pipefail

readonly script_dir="${0:A:h}"
readonly spike_dir="${script_dir:h}"
readonly repo_root="${spike_dir:h}"
readonly source_corpus="${spike_dir}/corpus/measurement-corpus-v2.json"
readonly appendix="${repo_root}/docs/superpowers/plans/2026-07-12-continuous-batching-chunked-prefill.md"
readonly output="${1:-${spike_dir}/corpus/measurement-corpus-v3.json}"

readonly expected_source_sha256="db0e3e5ff26ba88cec0bca67dd84eac353d1d169e362eefc884c8df050dc74c2"
readonly expected_appendix_sha256="e6948d5b4b550221dc86aaa7b0be7151857706c08166066666cc4d6355983a81"
readonly old_entry_id="long-context-engineering-docs-16k-v1"
readonly new_entry_id="long-context-engineering-docs-24k-v2"

sha256() {
    /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

require_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256 "${path}")"
    if [[ "${actual}" != "${expected}" ]]; then
        print -u2 -- "source hash mismatch: ${path}"
        print -u2 -- "expected ${expected}"
        print -u2 -- "actual   ${actual}"
        exit 1
    fi
}

require_sha256 "${source_corpus}" "${expected_source_sha256}"
require_sha256 "${appendix}" "${expected_appendix_sha256}"

jq_bin="$(command -v jq)"
readonly jq_bin
readonly output_dir="${output:h}"
/bin/mkdir -p "${output_dir}"
temporary="$(/usr/bin/mktemp "${output_dir}/.measurement-corpus-v3.XXXXXX")"
readonly temporary
trap '/bin/rm -f "${temporary}"' EXIT

# jq programs are intentionally single-quoted so jq, rather than the shell, expands `$appendix`.
# shellcheck disable=SC2016
"${jq_bin}" --sort-keys --indent 2 \
    --rawfile appendix "${appendix}" \
    --arg oldID "${old_entry_id}" \
    --arg newID "${new_entry_id}" \
    '
    .corpusId = "measurement-corpus-v3"
    | .entries = [
        .entries[]
        | if .id == $oldID then
            .id = $newID
            | .text +=
                "\n\n---\n\n"
                + "# fast-mlx continuous-batching and chunked-prefill design record\n\n"
                + $appendix
          else .
          end
      ]
    ' "${source_corpus}" > "${temporary}"

old_count="$(
    # shellcheck disable=SC2016
    "${jq_bin}" --arg id "${old_entry_id}" \
        '[.entries[] | select(.id == $id)] | length' "${source_corpus}"
)"
readonly old_count
new_count="$(
    # shellcheck disable=SC2016
    "${jq_bin}" --arg id "${new_entry_id}" \
        '[.entries[] | select(.id == $id)] | length' "${temporary}"
)"
readonly new_count
if [[ "${old_count}" != "1" || "${new_count}" != "1" ]]; then
    print -u2 -- "deep-entry replacement was not one-for-one"
    exit 1
fi

"${jq_bin}" -e '
    .corpusId == "measurement-corpus-v3"
    and (.entries | length == 5)
    and ([.entries[] | select(.tag == "long-context")] | length == 2)
    and ([.entries[] | select(.tag == "prose" or .tag == "code")] | length == 3)
    ' "${temporary}" >/dev/null

/bin/mv "${temporary}" "${output}"
trap - EXIT
print -- "$(sha256 "${output}")  ${output}"
