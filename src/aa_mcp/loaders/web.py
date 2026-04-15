"""Web page loader using httpx + BeautifulSoup."""

from __future__ import annotations

from aa_mcp.loaders.base import DocumentLoader


class WebLoader(DocumentLoader):
    async def load(self, source: str) -> tuple[str, str, dict]:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(source, headers={"User-Agent": "argument-analysis-mcp/0.1"})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove nav, footer, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else source
        text = soup.get_text(separator="\n", strip=True)

        return text, title, {"url": source, "content_type": response.headers.get("content-type", "")}
