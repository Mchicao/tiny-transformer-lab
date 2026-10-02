param([string]$SessionId = ('prep-' + [guid]::NewGuid().ToString('N')))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $project
$env:BROWSER = '"C:\Program Files\imput\Helium\Application\chrome.exe" %s'
Write-Host "Colab session ID: $SessionId"
& '.cache/vendor/colab-mcp/.venv/Scripts/python.exe' scripts/agents/colab_session.py --session-id $SessionId
