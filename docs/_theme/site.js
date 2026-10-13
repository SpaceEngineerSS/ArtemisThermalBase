// ArtemisThermalBase docs: theme switch, contents drawer, copy buttons,
// "on this page" tracking and heading search. No dependencies.
(function () {
  "use strict";
  var root = document.documentElement;

  function currentTheme() {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  var themeBtn = document.querySelector(".theme");
  function labelTheme() { if (themeBtn) themeBtn.textContent = currentTheme() === "dark" ? "Light" : "Dark"; }
  labelTheme();
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("atb-theme", next); } catch (e) {}
    labelTheme();
  });

  var menu = document.querySelector(".menu");
  if (menu) menu.addEventListener("click", function () {
    var open = document.body.classList.toggle("nav-open");
    menu.setAttribute("aria-expanded", open ? "true" : "false");
    menu.textContent = open ? "Close" : "Contents";
  });

  document.querySelectorAll("pre").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.className = "copy"; btn.type = "button"; btn.textContent = "copy";
    btn.addEventListener("click", function () {
      var text = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "copied";
        setTimeout(function () { btn.textContent = "copy"; }, 1200);
      });
    });
    pre.appendChild(btn);
  });

  // Fade lazy figures in as they load.
  document.querySelectorAll(".prose figure img").forEach(function (img) {
    if (img.complete) img.classList.add("seen");
    else img.addEventListener("load", function () { img.classList.add("seen"); });
  });

  var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (links.length && "IntersectionObserver" in window) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting; });
      var first = links.find(function (a) { return visible[a.getAttribute("href").slice(1)]; });
      if (first) { links.forEach(function (a) { a.classList.toggle("active", a === first); }); }
    }, { rootMargin: "-60px 0px -65% 0px" });
    Object.keys(byId).forEach(function (id) { var h = document.getElementById(id); if (h) io.observe(h); });
  }

  // Heading search.
  var overlay = document.querySelector(".search");
  var input = overlay && overlay.querySelector("input");
  var list = overlay && overlay.querySelector(".results");
  var index = null;
  function load(cb) {
    if (index) return cb();
    fetch("assets/search.json").then(function (r) { return r.json(); })
      .then(function (data) { index = data; cb(); })
      .catch(function () { index = []; cb(); });
  }
  function render(q) {
    q = q.trim().toLowerCase();
    list.innerHTML = "";
    if (!q) return;
    var words = q.split(/\s+/);
    var hits = index.filter(function (it) {
      var hay = (it.t + " " + it.p).toLowerCase();
      return words.every(function (w) { return hay.indexOf(w) >= 0; });
    }).slice(0, 12);
    hits.forEach(function (it, i) {
      var li = document.createElement("li");
      if (i === 0) li.className = "on";
      var a = document.createElement("a");
      a.href = it.u; a.textContent = it.t;
      if (it.p) { var s = document.createElement("span"); s.textContent = it.p; a.appendChild(s); }
      li.appendChild(a); list.appendChild(li);
    });
  }
  function open() { if (!overlay) return; load(function () { overlay.hidden = false; input.value = ""; list.innerHTML = ""; input.focus(); }); }
  function close() { if (overlay) overlay.hidden = true; }
  var openBtn = document.querySelector(".search-open");
  if (openBtn) openBtn.addEventListener("click", open);
  if (overlay) {
    overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { var a = list.querySelector("a"); if (a) window.location.href = a.href; }
    });
  }
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && !/input|textarea/i.test(document.activeElement.tagName)) { e.preventDefault(); open(); }
    else if (e.key === "Escape") close();
  });
})();
