$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed (exit $LASTEXITCODE). See main.log." }
} finally {
    Pop-Location
}
