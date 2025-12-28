import collections
import datetime
import html
import json
import os
import re
import shutil

import mistletoe
from mistletoe.html_renderer import HtmlRenderer


class Renderer(HtmlRenderer):
    def render_heading(self, token):
        template = '<h{level} id="{id}">{inner}</h{level}>'
        inner = self.render_inner(token)
        # clean ids: remove special chars, replace spaces with hyphens
        inner_id = re.sub(r'[^\w\s-]', '', inner.lower())
        inner_id = re.sub(r'[-\s]+', '-', inner_id).strip('-')
        return template.format(level=token.level, inner=inner, id=inner_id)

    def render_raw_text(self, token):
        content = token.content
        url_pattern = r'https?://[^\s<>"\'()]+(?:[^\s<>"\'().,;:!?\]])'
        parts = []
        last_end = 0
        for match in re.finditer(url_pattern, content):
            # add escaped text before URL
            parts.append(self.escape_html_text(content[last_end:match.start()]))
            # add unescaped link
            url = match.group(0)
            parts.append(f'<a href="{url}">{url}</a>')
            last_end = match.end()
        # add remaining escaped text
        parts.append(self.escape_html_text(content[last_end:]))
        return ''.join(parts)
    
    def render_link(self, token):
        template = '<a href="{target}"{title}{attrs}>{inner}</a>'
        target = self.escape_url(token.target)
        if token.title:
            title = ' title="{}"'.format(html.escape(token.title))
        else:
            title = ''
        # open external links in new tab
        attrs = ''
        if target.startswith(('http://', 'https://')):
            attrs = ' target="_blank" rel="noopener noreferrer"'
        inner = self.render_inner(token)
        return template.format(target=target, title=title, attrs=attrs, inner=inner)
    
    def render_image(self, token):
        template = '<img src="{}" alt="{}"{} loading="lazy" />'
        if token.title:
            title = ' title="{}"'.format(html.escape(token.title))
        else:
            title = ''
        return template.format(token.src, self.render_to_plain(token), title)
    
    def render_table(self, token):
        inner = super().render_table(token)
        return f'<div class="scrollable-table">{inner}</div>'

    def render_block_code(self, token):
        template = '<div class="code-block-wrapper"><button class="copy-code-btn" onclick="copyCode(this)">Copy</button><pre><code{attr}>{inner}</code></pre></div>'
        if token.language:
            attr = ' class="{}"'.format('language-{}'.format(html.escape(token.language)))
        else:
            attr = ''
        inner = self.escape_html_text(token.children[0].content if token.children else '')
        return template.format(attr=attr, inner=inner)

    def render_quote(self, token):
        alert_types = {
            '[!NOTE]': ('note', '💡', 'Note'),
            '[!TIP]': ('tip', '💡', 'Tip'),
            '[!IMPORTANT]': ('important', '❗', 'Important'),
            '[!WARNING]': ('warning', '⚠️', 'Warning'),
            '[!CAUTION]': ('caution', '🔥', 'Caution')
        }

        # check if first child is a paragraph with alert marker
        if token.children and hasattr(token.children[0], 'children'):
            first_child = token.children[0]
            if first_child.children and hasattr(first_child.children[0], 'content'):
                first_text = first_child.children[0].content.strip()                
                for marker, (alert_class, icon, label) in alert_types.items():
                    if first_text.startswith(marker):
                        first_child.children[0].content = first_child.children[0].content.replace(marker, '', 1).strip()
                        self._suppress_ptag_stack.append(False)
                        inner = '\n'.join([self.render(child) for child in token.children])
                        self._suppress_ptag_stack.pop()
                        return f'<div class="alert alert-{alert_class}"><div class="alert-title">{icon} {label}</div>{inner}</div>'
        
        # default blockquote rendering
        elements = ['<blockquote>']
        self._suppress_ptag_stack.append(False)
        elements.extend([self.render(child) for child in token.children])
        self._suppress_ptag_stack.pop()
        elements.append('</blockquote>')
        return '\n'.join(elements)


def simple_md_to_html(md):
    lines = md.split("\n")
    html = []
    in_code = False
    code_lang = ""
    in_list = False
    list_level = 0
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if line.startswith("```"):
            if in_code:
                html.append("</code></pre>")
                in_code = False
            else:
                code_lang = line[3:].strip() or "text"
                html.append(f'<pre><code class="{code_lang}">')
                in_code = True
            continue
        if in_code:
            html.append(html.escape(line))
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        line = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', line)
        line = re.sub(r"!\[(.+?)\]\((.+?)\)", r'<img src="\2" alt="\1">', line)
        line = re.sub(r'(?<!")(\bhttps?://\S+)(?!")', r'<a href="\1">\1</a>', line)
        if re.match(r"^#{1,6} ", stripped):
            if in_list:
                html.append("</ul>" * list_level)
                in_list = False
                list_level = 0
            level = len(re.match(r"^#+", stripped).group())
            text = re.sub(r"^#{1,6} ", "", stripped)
            html.append(f"<h{level}>{text}</h{level}>")
        elif re.match(r"^\s*- ", line):
            current_level = (indent // 2) + 1
            if not in_list:
                html.append("<ul>")
                in_list = True
                list_level = 1
            while current_level > list_level:
                html.append("<ul>")
                list_level += 1
            while current_level < list_level:
                html.append("</ul>")
                list_level -= 1
            li_text = re.sub(r"^\s*- ", "", line).strip()
            html.append(f"<li>{li_text}</li>")
        else:
            if in_list:
                html.append("</ul>" * list_level)
                in_list = False
                list_level = 0
            if stripped:
                html.append(f"<p>{stripped}</p>")
    if in_code:
        html.append("</code></pre>")
    if in_list:
        html.append("</ul>" * list_level)
    return "\n".join(html)


def parse_front_matter(content):
    if not content.startswith("---\n"):
        return {}, content
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return {}, content
    fm_str = parts[1].strip()
    fm = json.loads(fm_str)
    body = parts[2]
    return fm, body


def get_excerpt(body):
    paras = re.split(r"\n\s*\n", body.strip())
    return (paras[0][:150] + "...") if paras else ""


def collect_posts(src_dir):
    posts = []
    for root, dirs, files in os.walk(src_dir):
        if "README.md" not in files:
            continue
        path = os.path.join(root, "README.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        fm, body = parse_front_matter(content)
        if "title" not in fm:
            continue
        rel_path = os.path.relpath(root, src_dir)
        category_folder = rel_path.split("/")[0]
        category = category_folder.split("_")[1].capitalize()
        date_str = fm.get("date")
        date = date_str.split(" - ")[0] if " - " in date_str else date_str
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        posts.append(
            {
                "title": fm["title"],
                "url": f"/blog/{rel_path}/index.html",
                "date": date_str,
                "tags": fm.get("tags", []),
                "excerpt": get_excerpt(body),
                "fm": fm,
                "body": body,
                "root": root,
                "rel_path": rel_path,
                "category": category,
                "date_obj": date_obj,
            }
        )
    posts.sort(key=lambda p: p["date_obj"], reverse=True)
    return posts


def collect_news(src_file):
    with open(src_file, "r") as f:
        news = json.load(f)
    parsed_news = []
    for date_str, n in news.items():
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        parsed_news.append({
            "title": n,
            "date": date_str,
            "date_obj": date_obj,
        })
    parsed_news.sort(key=lambda p: p["date_obj"], reverse=True)
    return parsed_news


def load_template(template_dir, file_name):
    with open(os.path.join(template_dir, file_name), "r", encoding="utf-8") as f:
        return f.read()


def generate_post_html(template, post, authors: dict[str, str], css_path, js_path, rel_path):
    # body_html = simple_md_to_html(post["body"])
    # body_html = mistletoe.markdown(post["body"])
    with Renderer() as r:
        body_html = r.render(mistletoe.Document(post["body"]))

    tags_html = " ".join(f'<span class="tag">{t}</span>' for t in post["tags"])

    authors_list = post["fm"].get("authors", [])
    for i, author in enumerate(authors_list):
        if author not in authors:
            continue
        author_url = authors[author]
        authors_list[i] = f'<a href={author_url}>{author}</a>'
    authors_html = (
        f'<div class="post-authors">{" • ".join(authors_list)}</div>' if len(authors_list) > 0 else ""
    )

    links = []
    code_url = post["fm"].get("code")
    paper_url = post["fm"].get("paper")
    openreview_url = post["fm"].get("openreview")

    if code_url and str(code_url).strip() and str(code_url).lower() != "todo":
        links.append(
            '<a href="{0}" class="post-link code"><i class="fab fa-github"></i> Code</a>'.format(
                code_url
            )
        )
    if paper_url and str(paper_url).strip():
        links.append(
            '<a href="{0}" class="post-link paper"><i class="fas fa-file-pdf"></i> Paper</a>'.format(
                paper_url
            )
        )
    if openreview_url and str(openreview_url).strip():
        links.append(
            '<a href="{0}" class="post-link openreview"><i class="fas fa-external-link-alt"></i> OpenReview</a>'.format(
                openreview_url
            )
        )

    links_html = (
        '<div class="post-links">' + " ".join(links) + "</div>" if links else ""
    )

    return template.format(
        title=post["title"],
        css_path=css_path,
        date=post["date"],
        authors=authors_html,
        tags_html=tags_html,
        post_links=links_html,
        body_html=body_html,
        js_path=js_path,
        rel_path=rel_path,
    )


def generate_home_html(template, news: list[dict], posts: list[dict], css_path, js_path):
    combined_news_posts = []
    combined_news_posts.extend(news)
    combined_news_posts.extend(posts[:5])
    combined_news_posts.sort(key=lambda p: p["date_obj"], reverse=True)
    news_html = '<ul class="news-list">'
    for p in combined_news_posts:
        has_url = p.get("url", None) is not None
        if has_url:
            html = f'<li class="news-item"><span class="news-date">{p["date"]}</span><a href="{p["url"]}">[Blog] {p["title"]}</a></li>'
        else:
            html = f'<li class="news-item"><span class="news-date">{p["date"]}</span>{p["title"]}</li>'
        news_html += html
    news_html += '</ul>'
    return template.format(css_path=css_path, js_path=js_path, news_html=news_html)


def generate_blog_html(template, posts, css_path, js_path):
    posts_by_cat = collections.defaultdict(list)
    for p in posts:
        posts_by_cat[p["category"]].append(p)
    categories = reversed(["Seed", "Sprout", "Sapling", "Blossom"])
    cat_icons = {"Seed": "🌰", "Sprout": "🌱", "Sapling": "🪴", "Blossom": "🌸"}
    cat_html = ""
    for cat in categories:
        if cat not in posts_by_cat:
            continue
        post_list = ""
        for p in posts_by_cat[cat]:
            tags_html = " ".join(f'<span class="tag">{t}</span>' for t in p["tags"])
            post_list += f'<li class="post-item" data-tags="{" ".join(p["tags"])}"><div class="post-info"><div class="post-date">{p["date"]}</div><a href="{p["url"]}">{p["title"]}</a><div class="post-tags">{tags_html}</div></div></li>\n'
        icon = cat_icons.get(cat, "")
        details_open = "open" if cat == "Blossom" else ""
        cat_html += f'<details {details_open}><summary><span class="cat-icon">{icon}</span> {cat}</summary><ul>{post_list}</ul></details>\n'
    all_tags = set()
    for p in posts:
        all_tags.update(p["tags"])
    tag_list = "".join(
        f'<li><label><input type="checkbox" class="tag-filter" data-tag="{t}"> {t}</label></li>'
        for t in sorted(all_tags)
    )
    return template.format(
        css_path=css_path, cat_html=cat_html, tag_list=tag_list, js_path=js_path
    )


def main():
    template_dir = "templates"
    src_blog_dir = "src/blog"
    assets_dir = "assets"
    src_news_file = "src/news.json"
    src_authors_file = "src/authors.json"
    css_path = "/assets/style.css"
    js_path = "/assets/script.js"
    out_dir = "_site"
    assets_out_dir = os.path.join(out_dir, "assets")
    blog_out_dir = os.path.join(out_dir, "blog")
    post_template = load_template(template_dir, "post.html.template")
    home_template = load_template(template_dir, "home.html.template")
    blog_template = load_template(template_dir, "blog.html.template")

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(assets_out_dir, exist_ok=True)
    os.makedirs(blog_out_dir, exist_ok=True)
    shutil.copytree(assets_dir, assets_out_dir, dirs_exist_ok=True)

    posts = collect_posts(src_blog_dir)
    post_info_keys = {"title", "url", "date", "tags", "excerpt", "category"}
    with open(os.path.join(out_dir, "posts.json"), "w", encoding="utf-8") as f:
        post_info = [
            {k: v for k, v in p.items() if k in post_info_keys}
            for p in posts
        ]
        json.dump(post_info, f, indent=2)

    news = collect_news(src_news_file)

    with open(src_authors_file, "r") as f:
        authors = json.load(f)

    with open(os.path.join(assets_out_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(
            load_template(template_dir, "dark.css")
            + "\n"
            + load_template(template_dir, "light.css")
        )

    with open(os.path.join(assets_out_dir, "script.js"), "w", encoding="utf-8") as f:
        f.write(load_template(template_dir, "script.js"))

    home_html = generate_home_html(home_template, news, posts, css_path, js_path)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(home_html)
    
    with open(os.path.join(blog_out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            generate_blog_html(
                blog_template,
                posts,
                "/" + css_path.lstrip("/"),
                "/" + js_path.lstrip("/"),
            )
        )

    for post in posts:
        post_out_dir = os.path.join(out_dir, "blog", post["rel_path"])
        os.makedirs(post_out_dir, exist_ok=True)

        with open(os.path.join(post_out_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(
                generate_post_html(
                    post_template,
                    post,
                    authors,
                    "/" + css_path.lstrip("/"),
                    "/" + js_path.lstrip("/"),
                    post["rel_path"],
                )
            )

        for f in os.listdir(post["root"]):
            if f == "README.md":
                continue
            src = os.path.join(post["root"], f)
            dst = os.path.join(post_out_dir, f)
            if not os.path.isfile(src):
                continue
            with open(src, "rb") as s, open(dst, "wb") as d:
                d.write(s.read())


if __name__ == "__main__":
    main()
