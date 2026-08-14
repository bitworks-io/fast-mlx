(function () {
  "use strict";

  var FILTER_NAMES = ["model", "hardware", "decision"];
  var controls = document.querySelector("[data-benchmark-controls]");
  var cards = Array.prototype.slice.call(document.querySelectorAll("[data-benchmark-card]"));

  if (!controls || cards.length === 0) {
    return;
  }

  var selects = {};
  var validValues = {};
  var hasRequiredSelects = FILTER_NAMES.every(function (name) {
    var select = controls.querySelector('select[name="' + name + '"]');

    if (!select) {
      return false;
    }

    selects[name] = select;
    validValues[name] = new Set(Array.prototype.map.call(select.options, function (option) {
      return option.value;
    }));

    return true;
  });

  if (!hasRequiredSelects) {
    return;
  }

  var count = document.querySelector("[data-benchmark-count]");
  var empty = document.querySelector("[data-benchmark-empty]");
  var reset = document.querySelector("[data-benchmark-reset]");

  document.documentElement.classList.add("benchmark-enhanced");

  function validValueFor(name, value) {
    return validValues[name].has(value) ? value : "";
  }

  function activeFilters() {
    return FILTER_NAMES.reduce(function (filters, name) {
      filters[name] = validValueFor(name, selects[name].value);
      return filters;
    }, {});
  }

  function cardMatches(card, filters) {
    return FILTER_NAMES.every(function (name) {
      return filters[name] === "" || card.dataset[name] === filters[name];
    });
  }

  function updateQuery(filters) {
    var url = new URL(window.location.href);

    FILTER_NAMES.forEach(function (name) {
      if (filters[name]) {
        url.searchParams.set(name, filters[name]);
      } else {
        url.searchParams.delete(name);
      }
    });

    window.history.replaceState({}, "", url.pathname + url.search + url.hash);
  }

  function render() {
    var filters = activeFilters();
    var visibleCount = 0;

    cards.forEach(function (card) {
      var matches = cardMatches(card, filters);
      card.hidden = !matches;

      if (matches) {
        visibleCount += 1;
      }
    });

    if (count) {
      count.textContent = visibleCount + " of " + cards.length + " reviewed results shown.";
    }

    if (empty) {
      empty.hidden = visibleCount !== 0;
    }

    updateQuery(filters);
  }

  function applyParams() {
    var params = new URLSearchParams(window.location.search);

    FILTER_NAMES.forEach(function (name) {
      selects[name].value = validValueFor(name, params.get(name) || "");
    });
  }

  controls.addEventListener("change", function (event) {
    if (FILTER_NAMES.indexOf(event.target.name) !== -1) {
      render();
    }
  });

  function resetFilters() {
    FILTER_NAMES.forEach(function (name) {
      selects[name].value = "";
    });
    render();
  }

  controls.addEventListener("reset", function (event) {
    event.preventDefault();
    resetFilters();
  });

  if (reset) {
    reset.addEventListener("click", function (event) {
      event.preventDefault();
      resetFilters();
    });
  }

  applyParams();
  render();
}());
