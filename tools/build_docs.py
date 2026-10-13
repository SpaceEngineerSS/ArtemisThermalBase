"""Build the documentation site from docs/*.md into docs/_site/.

    python tools/build_docs.py            # build
    python tools/build_docs.py --strict   # also fail on broken links/anchors
    python tools/build_docs.py --serve    # build, then serve on localhost:8000

No theme framework: pages are rendered with markdown-it, laid out with one
Jinja template (docs/_theme/page.html) and styled by docs/_theme/site.css,
whose colour tokens come from artemis_thermal/style.py like the app's.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
THEME = DOCS / "_theme"
SITE = DOCS / "_site"
REPO_URL = "https://github.com/SpaceEngineerSS/ArtemisThermalBase"
sys.path.insert(0, str(ROOT))


@dataclass
class Page:
    """One source document."""

    source: Path
    slug: str
    title: str
    nav: str
    order: int
    kicker: str
    description: str
    body_md: str
    html: str = ""
    toc: list[tuple[int, str, str]] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[end + 5 :]


def collect() -> list[Page]:
    """Read every page: docs/*.md plus the root CHANGELOG."""
    sources = sorted(DOCS.glob("*.md")) + sorted((DOCS / "architecture").glob("*.md"))
    sources.append(ROOT / "CHANGELOG.md")
    pages = []
    for path in sources:
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        if path.name == "CHANGELOG.md":
            meta = {"title": "Changelog", "nav": "Changelog", "order": "90", "kicker": "",
                    "description": "What changed in each release."}
            body = re.sub(r"^# .*\n", "", body, count=1)
        slug = "index" if path.stem == "index" else path.stem.lower().replace("_", "-")
        if path.parent.name == "architecture":
            slug = f"adr-{slug.split('-', 2)[1]}" if slug.startswith("adr-") else slug
        title = meta.get("title") or _first_heading(body) or path.stem
        if meta.get("title") is None:
            body = re.sub(r"^# .*\n", "", body, count=1)
        pages.append(Page(
            source=path, slug=slug, title=title, nav=meta.get("nav", ""),
            order=int(meta.get("order", "50")), kicker=meta.get("kicker", ""),
            description=meta.get("description", ""), body_md=body,
        ))
    return sorted(pages, key=lambda p: (p.order, p.slug))


def _first_heading(text: str) -> str | None:
    match = re.search(r"^# (.+)$", text, re.M)
    return match.group(1).strip() if match else None


def _slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", text).strip("-")


def renderer():
    """markdown-it configured for our pages."""
    from markdown_it import MarkdownIt
    from mdit_py_plugins.anchors import anchors_plugin
    from mdit_py_plugins.deflist import deflist_plugin
    from mdit_py_plugins.dollarmath import dollarmath_plugin
    from mdit_py_plugins.footnote import footnote_plugin

    md = (
        MarkdownIt("commonmark", {"typographer": True, "html": True})
        .enable(["table", "strikethrough", "replacements", "smartquotes"])
        .use(dollarmath_plugin, double_inline=True)
        .use(anchors_plugin, min_level=2, max_level=3, slug_func=_slugify)
        .use(footnote_plugin)
        .use(deflist_plugin)
    )
    return md


_LINK = re.compile(r'(href|src)="([^"]+)"')


def _rewrite_links(page: Page, text: str, slugs: dict[str, str]) -> str:
    def fix(match: re.Match[str]) -> str:
        attr, url = match.groups()
        if url.startswith(("http://", "https://", "mailto:", "#", "data:")):
            return match.group(0)
        path, _, anchor = url.partition("#")
        base = page.source.parent
        target = (base / path).resolve() if path else page.source.resolve()
        if path.endswith(".md"):
            slug = slugs.get(str(target))
            if slug is not None:
                new = f"{slug}.html" + (f"#{anchor}" if anchor else "")
                return f'{attr}="{new}"'
        if path.startswith("img/") or "/img/" in path:
            rel = path.split("img/", 1)[1]
            return f'{attr}="img/{rel}"'
        try:
            rel_repo = target.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        kind = "tree" if target.is_dir() else "blob"
        return f'{attr}="{REPO_URL}/{kind}/main/{rel_repo}' + (f"#{anchor}" if anchor else "") + '"'

    return _LINK.sub(fix, text)


_IMG_P = re.compile(r'<p><img src="([^"]+)" alt="([^"]*)"(?: title="([^"]*)")? ?/?></p>')


def _figures(text: str) -> str:
    count = 0

    def figure(match: re.Match[str]) -> str:
        nonlocal count
        src, alt, title = match.groups()
        caption = title or ""
        if not caption:
            return f'<figure><img src="{src}" alt="{alt}" loading="lazy"></figure>'
        count += 1
        return (
            f'<figure id="fig-{count}"><img src="{src}" alt="{alt}" loading="lazy">'
            f'<figcaption><span class="figno">Fig. {count}</span> {caption}</figcaption></figure>'
        )

    return _IMG_P.sub(figure, text)


def _tables(text: str) -> str:
    return text.replace("<table>", '<div class="table"><table>').replace(
        "</table>", "</table></div>"
    )


def _math(text: str) -> str:
    text = re.sub(r'<div class="math block">(.*?)</div>',
                  lambda m: f'<div class="math display">\\[{m.group(1)}\\]</div>', text, flags=re.S)
    return re.sub(r'<span class="math inline">(.*?)</span>',
                  lambda m: f'<span class="math">\\({m.group(1)}\\)</span>', text, flags=re.S)


def _toc(text: str) -> list[tuple[int, str, str]]:
    found = re.findall(r'<h([23]) id="([^"]+)">(.*?)</h\1>', text)
    return [(int(level), anchor, re.sub(r"<[^>]+>", "", title)) for level, anchor, title in found]


def css() -> str:
    """Site stylesheet: tokens from the shared style module + layout."""
    from artemis_thermal import style

    def tokens(p: style.Palette) -> str:
        return "\n".join(f"  --{name.replace('_', '-')}: {value};"
                         for name, value in vars(p).items())

    temp = ", ".join(style.TEMPERATURE_STOPS)
    head = (
        f":root {{\n{tokens(style.LIGHT)}\n  --ramp: linear-gradient(90deg, {temp});\n"
        "  color-scheme: light;\n}\n"
        f"@media (prefers-color-scheme: dark) {{\n  :root:not([data-theme=\"light\"]) {{\n"
        f"{tokens(style.DARK)}\n    color-scheme: dark;\n  }}\n}}\n"
        f":root[data-theme=\"dark\"] {{\n{tokens(style.DARK)}\n  color-scheme: dark;\n}}\n"
    )
    return head + (THEME / "site.css").read_text(encoding="utf-8")


def build(strict: bool = False) -> list[str]:
    """Render the site; return a list of problems found."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from artemis_thermal import __version__

    pages = collect()
    slugs = {str(p.source.resolve()): p.slug for p in pages}
    md = renderer()
    for page in pages:
        rendered = md.render(page.body_md)
        rendered = _rewrite_links(page, rendered, slugs)
        rendered = _figures(_tables(_math(rendered)))
        page.html = rendered
        page.toc = _toc(rendered)
        page.ids = set(re.findall(r'id="([^"]+)"', rendered))

    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "assets").mkdir(parents=True)
    shutil.copytree(THEME / "fonts", SITE / "assets" / "fonts")
    shutil.copytree(THEME / "vendor", SITE / "assets" / "vendor")
    shutil.copy(THEME / "site.js", SITE / "assets" / "site.js")
    shutil.copy(ROOT / "artemis_thermal" / "assets" / "mark.svg", SITE / "assets" / "mark.svg")
    (SITE / "assets" / "site.css").write_text(css(), encoding="utf-8")
    if (DOCS / "img").is_dir():
        shutil.copytree(DOCS / "img", SITE / "img")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    env = Environment(loader=FileSystemLoader(THEME), autoescape=select_autoescape(["html"]))
    template = env.get_template("page.html")
    nav = [p for p in pages if p.nav]
    adrs = [p for p in pages if p.slug.startswith("adr-")]
    search = []
    for page in pages:
        out = template.render(page=page, nav=nav, adrs=adrs, version=__version__,
                              repo=REPO_URL, has_math='class="math' in page.html,
                              source_rel=page.source.relative_to(ROOT).as_posix())
        (SITE / f"{page.slug}.html").write_text(out, encoding="utf-8")
        for level, anchor, title in page.toc:
            search.append({"t": title, "p": page.title, "u": f"{page.slug}.html#{anchor}",
                           "l": level})
        search.append({"t": page.title, "p": "", "u": f"{page.slug}.html", "l": 1})
    (SITE / "assets" / "search.json").write_text(json.dumps(search, ensure_ascii=False),
                                                 encoding="utf-8")
    return check_links(pages) if strict else []


def check_links(pages: list[Page]) -> list[str]:
    """Broken internal links, anchors and images."""
    problems = []
    ids = {f"{p.slug}.html": p.ids for p in pages}
    for page in pages:
        text = (SITE / f"{page.slug}.html").read_text(encoding="utf-8")
        for attr, url in _LINK.findall(page.html):
            if url.startswith(("http", "mailto:", "data:")):
                continue
            path, _, anchor = url.partition("#")
            path = path or f"{page.slug}.html"
            if attr == "src" or path.startswith("img/"):
                if not (SITE / path).is_file():
                    problems.append(f"{page.source.name}: missing image {url}")
                continue
            if path not in ids:
                problems.append(f"{page.source.name}: broken link {url}")
            elif anchor and anchor not in ids[path] and not anchor.startswith("fig-"):
                problems.append(f"{page.source.name}: missing anchor {url}")
        if "{{" in text:
            problems.append(f"{page.source.name}: unrendered template text")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args(argv)
    problems = build(strict=args.strict or args.serve)
    for problem in problems:
        print(html.unescape(problem))
    print(f"Built {len(list(SITE.glob('*.html')))} pages into {SITE.relative_to(ROOT)}")
    if problems and args.strict:
        return 1
    if args.serve:
        import functools
        import http.server

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
        print("Serving on http://localhost:8000 (Ctrl+C to stop)")
        http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
