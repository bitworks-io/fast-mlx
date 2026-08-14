(function () {
  "use strict";

  var MAX_QUERY_LENGTH = 120;
  var controls = document.querySelector("[data-research-controls]");
  var cards = Array.prototype.slice.call(document.querySelectorAll("[data-research-card]"));

  if (!controls || cards.length === 0) {
    return;
  }

  var query = controls.querySelector('input[name="q"]');
  var theme = controls.querySelector('select[name="theme"]');

  if (!query || !theme) {
    return;
  }

  var validThemes = new Set(Array.prototype.map.call(theme.options, function (option) {
    return option.value;
  }));
  var count = document.querySelector("[data-research-count]");
  var empty = document.querySelector("[data-research-empty]");
  var reset = document.querySelector("[data-research-reset]");

  document.documentElement.classList.add("research-enhanced");

  function normalizedQuery(value) {
    var raw = String(value || "");
    if (raw.length > MAX_QUERY_LENGTH) {
      return "";
    }
    return raw.trim().replace(/\s+/g, " ");
  }

  function validTheme(value) {
    return validThemes.has(value) ? value : "";
  }

  function activeFilters() {
    var filters = {
      q: normalizedQuery(query.value),
      theme: validTheme(theme.value),
    };

    query.value = filters.q;
    theme.value = filters.theme;
    return filters;
  }

  function cardMatches(card, filters) {
    var themeMatches = filters.theme === "" || card.dataset.theme === filters.theme;
    var searchText = String(card.dataset.search || "").toLowerCase();
    var queryMatches = filters.q === "" || searchText.indexOf(filters.q.toLowerCase()) !== -1;
    return themeMatches && queryMatches;
  }

  function updateQuery(filters) {
    var url = new URL(window.location.href);

    if (filters.q) {
      url.searchParams.set("q", filters.q);
    } else {
      url.searchParams.delete("q");
    }
    if (filters.theme) {
      url.searchParams.set("theme", filters.theme);
    } else {
      url.searchParams.delete("theme");
    }

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
      count.textContent = "Showing " + visibleCount + " of " + cards.length + " reviewed notes.";
    }
    if (empty) {
      empty.hidden = visibleCount !== 0;
    }
    updateQuery(filters);
  }

  function applyParams() {
    var params = new URLSearchParams(window.location.search);
    query.value = normalizedQuery(params.get("q") || "");
    theme.value = validTheme(params.get("theme") || "");
  }

  function resetFilters() {
    query.value = "";
    theme.value = "";
    render();
  }

  controls.addEventListener("input", function (event) {
    if (event.target.name === "q") {
      render();
    }
  });

  controls.addEventListener("change", function (event) {
    if (event.target.name === "theme") {
      render();
    }
  });

  controls.addEventListener("submit", function (event) {
    event.preventDefault();
    render();
  });

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
