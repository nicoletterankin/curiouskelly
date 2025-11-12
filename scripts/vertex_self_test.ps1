param()
$ErrorActionPreference = "Stop"
$python = "python"
& $python tools/vertex_self_test.py
if ($LASTEXITCODE -ne 0) { throw "Vertex self-test failed" }


