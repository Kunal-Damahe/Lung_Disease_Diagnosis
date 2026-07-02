$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$latestModel = Get-ChildItem -Path (Join-Path $projectRoot "artifacts") `
    -Recurse -Filter "model.pt" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latestModel) {
    throw "No trained model was found under artifacts. Run training first."
}

$deploymentDirectory = Join-Path $projectRoot "deployment"
New-Item -ItemType Directory -Path $deploymentDirectory -Force | Out-Null
$destination = Join-Path $deploymentDirectory "model.pt"
Copy-Item -LiteralPath $latestModel.FullName -Destination $destination -Force

Write-Output "Packaged model: $($latestModel.FullName)"
Write-Output "Docker model: $destination"
