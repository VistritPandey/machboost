from mcp.server import MCPServer


server = MCPServer("MachBoost test connector")


@server.tool(description="Echo text through a real stdio MCP connection.")
def echo(text: str) -> str:
    return f"echo:{text}"


if __name__ == "__main__":
    server.run(transport="stdio")
