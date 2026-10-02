[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$ColabArgs,
    [Parameter(ValueFromPipeline = $true)][string]$InputObject
)

begin {
    $ErrorActionPreference = 'Stop'
    $stdinLines = [System.Collections.Generic.List[string]]::new()
}

process {
    if ($PSBoundParameters.ContainsKey('InputObject')) {
        $stdinLines.Add($InputObject)
    }
}

end {
    if ($stdinLines.Count) {
        $stdinLines | & wsl.exe -d Ubuntu --exec sh -c 'exec "$HOME/.local/share/tiny-transformer-lab/colab-cli/colab" "$@"' colab-wsl @ColabArgs
    } else {
        & wsl.exe -d Ubuntu --exec sh -c 'exec "$HOME/.local/share/tiny-transformer-lab/colab-cli/colab" "$@"' colab-wsl @ColabArgs
    }
    exit $LASTEXITCODE
}
