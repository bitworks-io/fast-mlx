#!/usr/bin/env python3
"""Generate the swift-jinja <-> transformers chat-template conformance fixture.

Renders synthetic (license-clean, hand-written) templates with Python Jinja2 in an
environment that matches transformers' chat-template environment
(src/transformers/utils/chat_template_utils.py): ImmutableSandboxedEnvironment with
trim_blocks=True, lstrip_blocks=True, the loopcontrols extension, and the
transformers `tojson` filter (json.dumps with ensure_ascii=False, insertion order).

Usage:
    python3 scripts/gen_jinja_conformance_fixture.py [--swift-out PATH]

Output is deterministic for a given jinja2 version. It is a Swift file embedding the JSON
table (the SpikeServingAdaptersTests target declares no resources), and the table records the
jinja2 version and this generator's path.
"""
import argparse
import json
import pathlib

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR_PATH = "scripts/gen_jinja_conformance_fixture.py"
DEFAULT_SWIFT_OUT = ROOT / "spike/Tests/SpikeServingAdaptersTests/JinjaTransformersConformanceTable.swift"


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # Exact transformers definition.
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_env():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    return env


TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "city": {"type": "string", "description": "City name"},
            },
        },
    },
}
TOOL_CLOCK = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the time in a zone",
        "parameters": {
            "type": "object",
            "properties": {"zone": {"type": "string"}},
            "required": ["zone"],
        },
    },
}

# NOTE: no template below ends in a newline. Python Jinja2 drops one trailing newline
# (keep_trailing_newline=False); swift-jinja 2.5.1 keeps it. That is a separate, known
# divergence, pinned by its own Swift test, and is kept out of these rows on purpose.
CASES = [
    {
        "name": "comment-trim-both-sides",
        "template": "alpha\n\n{#- trimmed -#}\n\nbeta\n{%- if true %}\ngamma\n{%- endif %}\n",
        "context": {},
    },
    {
        "name": "comment-trim-left-only",
        "template": "one\n\n{#- left only #}\n\ntwo",
        "context": {},
    },
    {
        "name": "comment-trim-right-only",
        "template": "one\n\n{# right only -#}\n\ntwo",
        "context": {},
    },
    {
        "name": "comment-plain-control",
        "template": "one\n{# plain #}\ntwo\n  {# indented #}\nthree",
        "context": {},
    },
    {
        "name": "tojson-plain-insertion-order",
        "template": "{{ tool | tojson }}",
        "context": {"tool": TOOL_WEATHER},
    },
    {
        "name": "tojson-indent-4-nested-list",
        "template": "{{ tool | tojson(indent=4) }}",
        "context": {"tool": TOOL_WEATHER},
    },
    {
        "name": "tojson-non-ascii",
        "template": "{{ s | tojson }}|{{ d | tojson }}",
        "context": {"s": "caf\u00e9", "d": {"city": "\u65e5\u672c", "greeting": "na\u00efve"}},
    },
    {
        "name": "tojson-slash-and-quotes",
        "template": "{{ s | tojson }}",
        "context": {"s": "a/b \"quoted\" c\\d\ttab\nline"},
    },
    {
        "name": "tojson-numbers-bools-null",
        "template": "{{ xs | tojson }}",
        "context": {"xs": [1, 2.5, -3, 0, True, False, None, "x"]},
    },
    {
        "name": "tojson-empty-containers",
        "template": "{{ o | tojson }} {{ a | tojson }}",
        "context": {"o": {}, "a": []},
    },
    {
        "name": "tools-loop-two-tools",
        "template": "{%- for t in tools %}{{- t | tojson }}{{ \"\\n\" }}{%- endfor %}",
        "context": {"tools": [TOOL_WEATHER, TOOL_CLOCK]},
    },
    {
        "name": "tools-loop-indent-four",
        "template": "{%- for t in tools %}\n{{- t | tojson(indent=4) }}\n{%- endfor %}\n",
        "context": {"tools": [TOOL_WEATHER, TOOL_CLOCK]},
    },
    {
        "name": "assistant-tool-call-history",
        "template": (
            "{%- for m in messages %}\n"
            "{%- if m.tool_calls %}\n"
            "{%- for tc in m.tool_calls %}\n"
            "<call>{\"name\": \"{{ tc.function.name }}\", \"arguments\": {{ tc.function.arguments | tojson }}}</call>\n"
            "{%- endfor %}\n"
            "{%- else %}\n"
            "<{{ m.role }}>{{ m.content }}</{{ m.role }}>\n"
            "{%- endif %}\n"
            "{%- endfor %}\n"
        ),
        "context": {
            "messages": [
                {"role": "user", "content": "Weather in Z\u00fcrich?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"unit": "celsius", "city": "Z\u00fcrich", "days": [1, 2], "path": "a/b"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "7 C"},
            ]
        },
    },
    {
        "name": "bos-messages-loop-with-comment-between-turns",
        "template": (
            "{{- bos_token }}\n"
            "{%- for m in messages %}\n"
            "<|x|>{{ m['role'] }}\n"
            "\n"
            "{#- turn separator -#}\n"
            "\n"
            "{{ m['content'] | trim }}<|e|>\n"
            "{%- endfor %}\n"
            "{%- if add_generation_prompt %}\n"
            "<|x|>assistant\n"
            "\n"
            "{%- endif %}"
        ),
        "context": {
            "bos_token": "<|b|>",
            "add_generation_prompt": True,
            "messages": [
                {"role": "system", "content": "  Be brief.  "},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ],
        },
    },
]


def build():
    env = make_env()
    rows = []
    for case in CASES:
        expected = env.from_string(case["template"]).render(**case["context"])
        rows.append(
            {
                "name": case["name"],
                "template": case["template"],
                "context": case["context"],
                "expected": expected,
            }
        )
    return {
        "schema": "jinja-transformers-conformance-v1",
        "generator": GENERATOR_PATH,
        "jinja2_version": jinja2.__version__,
        "environment": (
            "ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, "
            "extensions=[jinja2.ext.loopcontrols]); tojson = transformers json.dumps "
            "(ensure_ascii=False, indent=None, separators=None, sort_keys=False)"
        ),
        "cases": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swift-out", type=pathlib.Path, default=DEFAULT_SWIFT_OUT)
    args = ap.parse_args()
    doc = build()
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    assert '"""#' not in text
    swift = (
        "// GENERATED by scripts/gen_jinja_conformance_fixture.py -- do not edit.\n"
        "// The expected strings were rendered by Python jinja2 with the transformers chat-template\n"
        "// environment; this test target declares no resources, so the table is embedded here.\n"
        "enum JinjaTransformersConformanceTable {\n"
        '    static let json = #"""\n' + text + '"""#\n'
        "}\n"
    )
    args.swift_out.write_text(swift, encoding="utf-8")
    print(f"wrote {args.swift_out} ({len(doc['cases'])} cases, jinja2 {jinja2.__version__})")


if __name__ == "__main__":
    main()
