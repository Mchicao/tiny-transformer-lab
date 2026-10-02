param([string]$Uv = 'C:\Users\matia\AppData\Local\hermes\bin\uv.exe')
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $project
$env:UV_CACHE_DIR = Join-Path $project '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $project '.cache/python'
function Invoke-Uv {
    & $Uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed: $args" }
}
if (!(Test-Path '.venv/Scripts/python.exe')) { Invoke-Uv venv --python 3.12 .venv }
Invoke-Uv pip install --python .venv/Scripts/python.exe --require-hashes -r configs/requirements-amd.lock.txt --index-url https://stable.repo.amd.com/rocm/whl-next/
Invoke-Uv pip install --python .venv/Scripts/python.exe --require-hashes -r configs/requirements-common.lock.txt
$vendor = Join-Path $project '.cache/vendor/colab-mcp'
$commit = 'b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1'
if (!(Test-Path $vendor)) {
    git clone https://github.com/googlecolab/colab-mcp.git $vendor
    if ($LASTEXITCODE -ne 0) { throw 'Official MCP clone failed' }
    git -C $vendor checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned MCP checkout failed' }
}
if ((git -C $vendor rev-parse HEAD) -ne $commit) { throw 'MCP revision differs; preserve checkout and investigate' }
Invoke-Uv sync --frozen --no-dev --project $vendor --python 3.14
Invoke-Uv pip check --python .venv/Scripts/python.exe
Write-Output 'Entornos preparados. Este script no ejecuta entrenamientos ni benchmarks.'
