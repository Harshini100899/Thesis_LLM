# Find coder.exe in WinGet packages
$coderExe = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "coder.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName

if (-not $coderExe) {
    Write-Error "coder.exe not found in WinGet packages. Install it first with: winget install Coder.Coder"
    exit 1
}

$coderDir = Split-Path $coderExe
Write-Host "Found coder.exe at: $coderExe"

# Check if already in PATH
if ($env:Path -split ";" -contains $coderDir) {
    Write-Host "Already in PATH."
} else {
    $env:Path += ";$coderDir"
    [Environment]::SetEnvironmentVariable("Path", $env:Path, "User")
    Write-Host "Added $coderDir to user PATH."
}

# Verify version and login status
& "$coderExe" version
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. coder login https://coder.compute.isst.fraunhofer.de"
Write-Host "  2. coder config-ssh --yes"
