param(
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"
$scriptRoot = $PSScriptRoot

function Resolve-RepoRoot {
    param([string]$InputRoot)

    if ([string]::IsNullOrWhiteSpace($InputRoot)) {
        return (Resolve-Path (Join-Path $scriptRoot "..")).Path
    }

    return (Resolve-Path $InputRoot).Path
}

$resolvedRepoRoot = Resolve-RepoRoot -InputRoot $RepoRoot
$mcpRoot = Join-Path $resolvedRepoRoot "mcp_server"

if (-not (Test-Path $mcpRoot)) {
    throw "mcp_server directory not found: $mcpRoot"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not available in PATH."
}

Push-Location $mcpRoot
try {
    Write-Host "[start-mcp-stdio] Syncing dependencies..."
    uv sync | Out-Host

    Write-Host "[start-mcp-stdio] Starting MCP server over stdio..."
    uv run sts2-mcp-server
}
finally {
    Pop-Location
}
