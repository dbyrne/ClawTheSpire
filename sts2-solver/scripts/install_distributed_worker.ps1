param(
    [Parameter(Mandatory = $true)]
    [string]$Coordinator,

    [string]$Experiment = "",
    [string]$WorkerId = $env:COMPUTERNAME,
    [string]$TaskName = "STS2 Distributed Worker",
    [double]$LeaseSeconds = 240,
    [string]$Python = "",
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Python) {
    $Python = Join-Path $SolverRoot ".venv\Scripts\python.exe"
}
$Python = (Resolve-Path $Python).Path

& $Python -c "import sts2_engine, sts2_solver" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "This Python cannot import sts2_engine and sts2_solver. Install the solver and run maturin develop first."
}

$workerArgs = @(
    "-m", "sts2_solver.betaone.distributed_worker",
    "--coordinator", $Coordinator,
    "--worker-id", $WorkerId,
    "--lease-s", "$LeaseSeconds"
)
if ($Experiment) {
    $workerArgs += @("--experiment", $Experiment)
}
$argString = ($workerArgs | ForEach-Object {
    if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join " "

Write-Host "Worker command:"
Write-Host "  $Python $argString"
if ($PrintOnly) {
    exit 0
}

$logsDir = Join-Path $SolverRoot "logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
$logFile = Join-Path $logsDir "distributed_worker.log"
$launcher = Join-Path $logsDir "distributed_worker_launcher.cmd"
@"
@echo off
cd /d "$SolverRoot"
"$Python" $argString >> "$logFile" 2>&1
"@ | Set-Content -Path $launcher -Encoding ASCII

$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" -Argument "/c `"$launcher`"" -WorkingDirectory $SolverRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Polls the STS2 companion coordinator for distributed self-play shards." `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Worker log: $logFile"
