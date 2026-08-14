"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");

const productionScript = process.argv[2];
assert.ok(productionScript, "expected the benchmark explorer script path");
const researchScript = process.argv[3];
assert.ok(researchScript, "expected the research explorer script path");

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

const researchHandlers = {};
const researchResetHandlers = {};
const queryInput = {name: "q", value: ""};
const themeSelect = select("theme", ["", "Theme A", "Theme B"]);
const researchControls = {
  querySelector: function (selector) {
    if (selector === 'input[name="q"]') {
      return queryInput;
    }
    if (selector === 'select[name="theme"]') {
      return themeSelect;
    }
    return null;
  },
  addEventListener: function (name, handler) { researchHandlers[name] = handler; },
};
const researchCards = [
  {
    dataset: {
      search: "The fastest request wasn't the fastest service Theme A",
      theme: "Theme A",
    },
    hidden: false,
  },
  {
    dataset: {
      search: "The proof did not end when the timer did Theme B",
      theme: "Theme B",
    },
    hidden: false,
  },
];
const researchCount = {textContent: ""};
const researchEmpty = {hidden: true};
const researchReset = {
  addEventListener: function (name, handler) { researchResetHandlers[name] = handler; },
};
const researchRootClasses = new Set();
const overlongQuery = "x".repeat(121);
const researchLocation = {
  href: "https://example.test/research/?q=" + overlongQuery + "&theme=invalid#notes",
  search: "?q=" + overlongQuery + "&theme=invalid",
};
const researchWindowFixture = {
  location: researchLocation,
  history: {
    replaceState: function (_state, _title, relativeUrl) {
      const next = new URL(relativeUrl, researchLocation.href);
      researchLocation.href = next.href;
      researchLocation.search = next.search;
    },
  },
};
const researchDocumentFixture = {
  documentElement: {
    classList: {
      add: function (name) { researchRootClasses.add(name); },
    },
  },
  querySelector: function (selector) {
    return {
      "[data-research-controls]": researchControls,
      "[data-research-count]": researchCount,
      "[data-research-empty]": researchEmpty,
      "[data-research-reset]": researchReset,
    }[selector] || null;
  },
  querySelectorAll: function (selector) {
    return selector === "[data-research-card]" ? researchCards : [];
  },
};

global.window = researchWindowFixture;
global.document = researchDocumentFixture;
require(path.resolve(researchScript));

assert.equal(researchRootClasses.has("research-enhanced"), true);
assert.equal(queryInput.value, "", "overlong URL query must be dropped");
assert.equal(themeSelect.value, "", "invalid URL theme must be ignored");
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [false, false]);
assert.equal(researchCount.textContent, "Showing 2 of 2 reviewed notes.");
assert.equal(researchEmpty.hidden, true);
assert.equal(researchLocation.href, "https://example.test/research/#notes");

queryInput.value = "service";
researchHandlers.input({target: queryInput});
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [false, true]);
assert.equal(researchCount.textContent, "Showing 1 of 2 reviewed notes.");
assert.equal(new URL(researchLocation.href).searchParams.get("q"), "service");

themeSelect.value = "Theme B";
researchHandlers.change({target: themeSelect});
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [true, true]);
assert.equal(researchCount.textContent, "Showing 0 of 2 reviewed notes.");
assert.equal(researchEmpty.hidden, false);

queryInput.value = "  PROOF   DID ";
researchHandlers.input({target: queryInput});
assert.equal(queryInput.value, "PROOF DID");
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [true, false]);
assert.equal(researchCount.textContent, "Showing 1 of 2 reviewed notes.");
assert.equal(new URL(researchLocation.href).searchParams.get("q"), "PROOF DID");
assert.equal(new URL(researchLocation.href).searchParams.get("theme"), "Theme B");

queryInput.value = '<img src=x onerror="alert(1)">';
researchHandlers.input({target: queryInput});
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [true, true]);
assert.equal(new URL(researchLocation.href).searchParams.get("q"), '<img src=x onerror="alert(1)">');

queryInput.value = "y".repeat(121);
researchHandlers.input({target: queryInput});
assert.equal(queryInput.value, "");
assert.equal(new URL(researchLocation.href).searchParams.has("q"), false);
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [true, false]);

let submitPrevented = false;
researchHandlers.submit({preventDefault: function () { submitPrevented = true; }});
assert.equal(submitPrevented, true);

let researchResetPrevented = false;
researchHandlers.reset({preventDefault: function () { researchResetPrevented = true; }});
assert.equal(researchResetPrevented, true);
assert.equal(queryInput.value, "");
assert.equal(themeSelect.value, "");
assert.deepEqual(researchCards.map(function (card) { return card.hidden; }), [false, false]);
assert.equal(researchCount.textContent, "Showing 2 of 2 reviewed notes.");
assert.equal(researchEmpty.hidden, true);
assert.equal(researchLocation.href, "https://example.test/research/#notes");
assert.equal(typeof researchResetHandlers.click, "function");

console.log("research explorer runtime checks passed");
