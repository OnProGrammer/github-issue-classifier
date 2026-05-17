import re
import logging

from bs4 import BeautifulSoup

from scraper.config import ALL_LABEL_SUFFIXES

logger = logging.getLogger(__name__)


def has_no_results(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    h3 = soup.find("h3", class_="blankslate-heading")
    if h3 is None:
        return False
    return "No results" in h3.get_text(strip=True)


def parse_issue_list(html: str, repo: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    result = []

    for a in soup.select('a[data-testid="issue-pr-title-link"]'):
        href = a.get("href", "")
        if f"{repo}/issues/" not in href:
            continue

        title = a.get_text(strip=True)
        if not title:
            continue

        li = a.find_parent("li") or a.find_parent("div")
        if li is None:
            continue

        labels = _extract_labels(li)
        issue_id = _extract_issue_id(li, href)
        if issue_id == -1:
            continue

        result.append({
            "issue_id": issue_id,
            "url": f"https://github.com{href}",
            "title_raw": title,
            "labels_raw": labels,
        })

    seen: set[int] = set()
    unique = []
    for item in result:
        if item["issue_id"] not in seen:
            seen.add(item["issue_id"])
            unique.append(item)

    logger.debug("parsed %d unique issues from list page", len(unique))
    return unique


def parse_issue_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body_div = soup.find("div", class_="markdown-body")
    if body_div is None:
        return ""
    return body_div.get_text("\n", strip=True)


def parse_issue_meta(html: str) -> dict[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    state = None
    for span in soup.find_all("span"):
        text = span.get_text(strip=True)
        if text == "Open" and state is None:
            state = "open"
        elif text.startswith("Closed") and state is None:
            state = "closed"

    created_at = None
    rt = soup.find("relative-time")
    if rt is not None and rt.has_attr("datetime"):
        created_at = rt["datetime"]

    return {"state": state, "created_at": created_at}


def build_text_for_model(title: str, body: str) -> str:
    title_clean = title.strip()
    body_clean = body.strip()
    if not body_clean:
        return f"[TITLE] {title_clean}"
    return f"[TITLE] {title_clean} [BODY] {body_clean}"


def classify_target(labels: list[str]) -> str | None:
    candidates: set[str] = set()
    for label in labels:
        if label.startswith("kind/"):
            suffix = label.split("/", 1)[1]
            if suffix in ALL_LABEL_SUFFIXES:
                candidates.add(ALL_LABEL_SUFFIXES[suffix])

    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _extract_labels(container) -> list[str]:
    labels = []
    label_container = container.select_one(
        "span.Title-module__trailingBadgesContainer--mijcn"
    )
    if label_container is not None:
        for span in label_container.select("span.Text__StyledText-sc-1klmep6-0"):
            text = span.get_text(strip=True)
            if text:
                labels.append(text)
    return labels


def _extract_issue_id(container, href: str) -> int:
    id_span = container.select_one("div.MainContent-module__container--NyRpm span")
    if id_span is not None:
        match = re.search(r"#(\d+)", id_span.get_text(strip=True))
        if match:
            return int(match.group(1))

    match = re.search(r"/issues/(\d+)", href)
    if match:
        return int(match.group(1))

    return -1
