$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

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

$pythonPath = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "Virtual environment was not found. Run .\setup_project.ps1 first."
}

$nodeRuntime = Get-NodeRuntime
if (-not $nodeRuntime) {
    throw "npm was not found. Install Node.js LTS, open a new PowerShell window, then rerun .\setup_project.ps1."
}

Write-Host "Starting FastAPI on http://localhost:8001 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$PSScriptRoot'; & '$pythonPath' -m uvicorn app.step12_fastapi:app --reload --port 8001"
)

Start-Sleep -Seconds 4

Write-Host "Starting React dashboard on http://localhost:5173 ..." -ForegroundColor Cyan
Set-Location (Join-Path $PSScriptRoot "frontend")

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    $env:ComSpec = Join-Path $env:SystemRoot "System32\cmd.exe"
    $env:comspec = $env:ComSpec
    $env:SHELL = $env:ComSpec
    $env:npm_config_script_shell = $env:ComSpec
    & $nodeRuntime.Node $nodeRuntime.NpmCli install --script-shell="$env:ComSpec"
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend dependency install failed."
    }
}

$env:ComSpec = Join-Path $env:SystemRoot "System32\cmd.exe"
$env:comspec = $env:ComSpec
$env:SHELL = $env:ComSpec
$env:npm_config_script_shell = $env:ComSpec
& $nodeRuntime.Node $nodeRuntime.NpmCli run dev --script-shell="$env:ComSpec"
if ($LASTEXITCODE -ne 0) {
    throw "React dashboard failed to start."
}
