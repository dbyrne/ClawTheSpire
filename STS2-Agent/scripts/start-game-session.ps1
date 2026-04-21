param(
    [string]$ExePath = "C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.exe",
    [int]$Attempts = 40,
    [int]$DelaySeconds = 2,
    [switch]$EnableDebugActions,
    [int]$ApiPort = 8080,
    [switch]$KeepExistingProcesses
)

$ErrorActionPreference = "Stop"

function Wait-ForHealth {
    param(
        [int]$MaxAttempts,
        [int]$SleepSeconds,
        [System.Diagnostics.Process]$Process,
        [string]$BaseUrl
    )

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        Start-Sleep -Seconds $SleepSeconds

        try {
            $response = Invoke-WebRequest -Uri ($BaseUrl.TrimEnd("/") + "/health") -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
        }

        if ($Process.HasExited) {
            throw "Game process exited before /health became ready."
        }
    }

    throw "Timed out waiting for /health."
}

function Wait-ForPortRelease {
    param(
        [int]$MaxAttempts,
        [int]$SleepSeconds,
        [int]$Port
    )

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        try {
            $listenerActive = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop).Count -gt 0
        } catch {
            $listenerActive = $false
        }

        if (-not $listenerActive) {
            return
        }

        Start-Sleep -Seconds $SleepSeconds
    }
}

$baseUrl = "http://127.0.0.1:$ApiPort"

if (-not $KeepExistingProcesses) {
    $existing = Get-Process -Name "SlayTheSpire2" -ErrorAction SilentlyContinue
    if ($existing) {
        Stop-Process -Id $existing.Id -Force
        Start-Sleep -Seconds 2
        Wait-ForPortRelease -MaxAttempts 10 -SleepSeconds 1 -Port $ApiPort
    }
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $ExePath
$startInfo.UseShellExecute = $false

if ($EnableDebugActions) {
    $startInfo.EnvironmentVariables["STS2_ENABLE_DEBUG_ACTIONS"] = "1"
} else {
    $startInfo.EnvironmentVariables.Remove("STS2_ENABLE_DEBUG_ACTIONS")
}

$startInfo.EnvironmentVariables["STS2_API_PORT"] = [string]$ApiPort

$proc = [System.Diagnostics.Process]::Start($startInfo)
Wait-ForHealth -MaxAttempts $Attempts -SleepSeconds $DelaySeconds -Process $proc -BaseUrl $baseUrl

[pscustomobject]@{
    pid = $proc.Id
    debug_actions_enabled = [bool]$EnableDebugActions
    api_port = $ApiPort
    base_url = $baseUrl
    health = "ready"
} | ConvertTo-Json -Compress
