$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Setting up Project 02 - AR Recovery Engine..." -ForegroundColor Cyan

$venvDir = ".venv312"
$venvPath = Join-Path $PSScriptRoot $venvDir

function Get-NodeRuntime {
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    $nodePath = if ($node) { $node.Source } else { $null }

    $commonNodePaths = @(
        "$env:ProgramFiles\nodejs\node.exe",
        "${env:ProgramFiles(x86)}\nodejs\node.exe"
    )

    foreach ($path in $commonNodePaths) {
        if (-not $nodePath -and $path -and (Test-Path $path)) {
            $nodePath = $path
        }
    }

    $commonNpmCliPaths = @(
        "$env:ProgramFiles\nodejs\node_modules\npm\bin\npm-cli.js",
        "${env:ProgramFiles(x86)}\nodejs\node_modules\npm\bin\npm-cli.js"
    )

    $npmCliPath = $null
    foreach ($path in $commonNpmCliPaths) {
        if (-not $npmCliPath -and $path -and (Test-Path $path)) {
            $npmCliPath = $path
        }
    }

    if ($nodePath -and $npmCliPath) {
        return @{ Node = $nodePath; NpmCli = $npmCliPath }
    }

    return $null
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    py -3.12 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create Python 3.12 virtual environment. Install Python 3.12, then rerun setup."
    }
}

$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Virtual environment Python was not found at $pythonPath"
}

$pythonVersionText = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$pythonMinor = & $pythonPath -c "import sys; print(sys.version_info.minor)"

if ([int]$pythonMinor -ge 13) {
    throw "This project is pinned for Python 3.11 or 3.12. Current virtual environment uses Python $pythonVersionText, which can force pandas to build from source. Recreate the environment with Python 3.12, then rerun setup."
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

Write-Host "Installing requirements..." -ForegroundColor Yellow
& $pythonPath -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Python requirements install failed."
}

$nodeRuntime = Get-NodeRuntime
if ($nodeRuntime) {
    Write-Host "Installing React frontend dependencies..." -ForegroundColor Yellow
    Set-Location (Join-Path $PSScriptRoot "frontend")
    $env:ComSpec = Join-Path $env:SystemRoot "System32\cmd.exe"
    $env:comspec = $env:ComSpec
    $env:SHELL = $env:ComSpec
    $env:npm_config_script_shell = $env:ComSpec
    & $nodeRuntime.Node $nodeRuntime.NpmCli install --script-shell="$env:ComSpec"
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend dependency install failed."
    }
    Set-Location $PSScriptRoot
} else {
    throw "npm was not found. Install Node.js LTS before running the React dashboard."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run .\run_pipeline.ps1 to rebuild outputs, or .\run_app.ps1 to start the React demo app." -ForegroundColor Cyan
Write-Host "Use $venvDir\Scripts\python.exe as the VS Code interpreter." -ForegroundColor Cyan
