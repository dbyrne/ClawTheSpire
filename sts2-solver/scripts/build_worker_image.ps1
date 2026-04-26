param(
    [Parameter(Mandatory = $true)]
    [string]$ImageRepository,

    [string]$TagPrefix = "",
    [switch]$Push,
    [switch]$EcrLogin,
    [string]$AwsRegion = ""
)

$ErrorActionPreference = "Stop"

$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $SolverRoot "..")).Path
$GitSha = (git -C $RepoRoot rev-parse HEAD).Trim()
$Branch = (git -C $RepoRoot branch --show-current).Trim()
if (-not $TagPrefix) {
    $TagPrefix = ($Branch -replace "[^A-Za-z0-9_.-]", "-")
}
$Image = "${ImageRepository}:${TagPrefix}-${GitSha}"

if ($EcrLogin) {
    if (-not $AwsRegion) {
        $AwsRegion = (& "C:\Program Files\Amazon\AWSCLIV2\aws.exe" configure get region).Trim()
        if (-not $AwsRegion) {
            $AwsRegion = "us-east-1"
        }
    }
    $Registry = $ImageRepository.Split("/")[0]
    & "C:\Program Files\Amazon\AWSCLIV2\aws.exe" ecr get-login-password --region $AwsRegion |
        docker login --username AWS --password-stdin $Registry
}

docker build `
    --build-arg "STS2_GIT_SHA=$GitSha" `
    --build-arg "STS2_IMAGE_SOURCE=$Branch" `
    -f (Join-Path $SolverRoot "Dockerfile.worker") `
    -t $Image `
    $RepoRoot

if ($Push) {
    docker push $Image
}

Write-Host "Worker image: $Image"
Write-Host "Git SHA: $GitSha"
