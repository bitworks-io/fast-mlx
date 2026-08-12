"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");

const productionScript = process.argv[2];
assert.ok(productionScript, "expected the benchmark explorer script path");

function select(name, values) {
  return {
    name: name,
    options: values.map(function (value) { return {value: value}; }),
    value: "",
  };
}

const selects = {
  model: select("model", ["", "Model A"]),
  hardware: select("hardware", ["", "Hardware A", "Hardware B"]),
  decision: select("decision", ["", "promoted-scoped", "shelved"]),
};
const handlers = {};
const resetHandlers = {};
const controls = {
  querySelector: function (selector) {
    const match = selector.match(/^select\[name="([a-z]+)"\]$/);
    return match ? selects[match[1]] : null;
  },
  addEventListener: function (name, handler) { handlers[name] = handler; },
};
const cards = [
  {dataset: {model: "Model A", hardware: "Hardware A", decision: "shelved"}, hidden: false},
  {dataset: {model: "Model A", hardware: "Hardware B", decision: "promoted-scoped"}, hidden: false},
];
const count = {textContent: ""};
const empty = {hidden: true};
const reset = {
  addEventListener: function (name, handler) { resetHandlers[name] = handler; },
};
const rootClasses = new Set();
const location = {
  href: "https://example.test/benchmarks/?hardware=invalid&decision=shelved#scope",
  search: "?hardware=invalid&decision=shelved",
};
const windowFixture = {
  location: location,
  history: {
    replaceState: function (_state, _title, relativeUrl) {
      const next = new URL(relativeUrl, location.href);
      location.href = next.href;
      location.search = next.search;
    },
  },
};
const documentFixture = {
  documentElement: {
    classList: {
      add: function (name) { rootClasses.add(name); },
    },
  },
  querySelector: function (selector) {
    return {
      "[data-benchmark-controls]": controls,
      "[data-benchmark-count]": count,
      "[data-benchmark-empty]": empty,
      "[data-benchmark-reset]": reset,
    }[selector] || null;
  },
  querySelectorAll: function (selector) {
    return selector === "[data-benchmark-card]" ? cards : [];
  },
};

global.window = windowFixture;
global.document = documentFixture;
require(path.resolve(productionScript));

assert.equal(rootClasses.has("benchmark-enhanced"), true);
assert.equal(selects.hardware.value, "", "invalid URL values must be ignored");
assert.equal(selects.decision.value, "shelved");
assert.deepEqual(cards.map(function (card) { return card.hidden; }), [false, true]);
assert.equal(count.textContent, "1 of 2 reviewed results shown.");
assert.equal(empty.hidden, true);
assert.equal(location.href, "https://example.test/benchmarks/?decision=shelved#scope");

selects.hardware.value = "Hardware B";
handlers.change({target: selects.hardware});
assert.deepEqual(cards.map(function (card) { return card.hidden; }), [true, true]);
assert.equal(count.textContent, "0 of 2 reviewed results shown.");
assert.equal(empty.hidden, false);
const zeroResultUrl = new URL(location.href);
assert.equal(zeroResultUrl.searchParams.get("hardware"), "Hardware B");
assert.equal(zeroResultUrl.searchParams.get("decision"), "shelved");
assert.equal(zeroResultUrl.hash, "#scope");

let resetPrevented = false;
handlers.reset({preventDefault: function () { resetPrevented = true; }});
assert.equal(resetPrevented, true);
assert.deepEqual(
  [selects.model.value, selects.hardware.value, selects.decision.value],
  ["", "", ""]
);
assert.deepEqual(cards.map(function (card) { return card.hidden; }), [false, false]);
assert.equal(count.textContent, "2 of 2 reviewed results shown.");
assert.equal(empty.hidden, true);
assert.equal(location.href, "https://example.test/benchmarks/#scope");
assert.equal(typeof resetHandlers.click, "function");

console.log("benchmark explorer runtime checks passed");
