param(
    [Parameter(Mandatory = $true)]
    [string]$AssemblyPath,

    [string[]]$Patterns = @(
        'ModManager',
        'ModInitializerAttribute',
        'CombatManager',
        'RunManager',
        'EnqueueManualPlay',
        'UsePotion'
    ),

    [int]$MinLength = 4
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $AssemblyPath)) {
    throw "Assembly not found: $AssemblyPath"
}

$text = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($AssemblyPath))
$regex = "[ -~]{$MinLength,}"
$matches = [regex]::Matches($text, $regex) | ForEach-Object { $_.Value }

$results = $matches |
    Where-Object {
        foreach ($pattern in $Patterns) {
            if ($_ -like "*$pattern*") {
                return $true
            }
        }
        return $false
    } |
    Sort-Object -Unique

$results
