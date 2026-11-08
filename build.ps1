# Build the package into dist/, and optionally upload it to PyPI.
#
#   .\build.ps1                 build and check dist/*.whl and dist/*.tar.gz
#   .\build.ps1 -Publish        build, then upload to PyPI
#   .\build.ps1 -Publish -Test  build, then upload to TestPyPI first
#
# Bump __version__ in dejadrew/__init__.py before each upload: PyPI never
# accepts the same version twice.

param(
    [switch]$Publish,
    [switch]$Test
)

Set-Location $PSScriptRoot

$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

& $python -m pip install --quiet --upgrade build twine
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Remove-Item -Recurse -Force dist, build, dejadrew.egg-info -ErrorAction SilentlyContinue

& $python -m build
if ($LASTEXITCODE) { exit $LASTEXITCODE }

& $python -m twine check dist/*
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if ($Publish) {
    if ($Test) {
        & $python -m twine upload --repository testpypi dist/*
    } else {
        & $python -m twine upload dist/*
    }
    exit $LASTEXITCODE
}

Write-Host "Built:" (Get-ChildItem dist | ForEach-Object Name)
