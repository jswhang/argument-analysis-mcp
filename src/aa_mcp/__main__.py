"""Entry point for `python -m aa_mcp` and the `aa-mcp-server` script."""

from __future__ import annotations


def main() -> None:
    from aa_mcp.server import mcp

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
