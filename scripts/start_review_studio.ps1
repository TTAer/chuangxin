$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
& $taskPython (Join-Path $taskRoot 'scripts\build_review_studio.py')
if ($LASTEXITCODE -ne 0) { throw 'Review build or evidence lock check failed.' }
$taskUrl = 'http://127.0.0.1:8767/review-studio/index.html?pass=A'
$taskRunning = $false
try {
    $taskResponse = Invoke-WebRequest -Uri $taskUrl -UseBasicParsing -TimeoutSec 3
    $taskRunning = $taskResponse.StatusCode -eq 200
} catch { }
if (-not $taskRunning) {
    $taskOutput = Join-Path $taskRoot 'output'
    $taskArguments = @('-m', 'http.server', '8767', '--bind', '127.0.0.1', '--directory', ('"' + $taskOutput + '"'))
    Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WindowStyle Hidden -WorkingDirectory $taskRoot
    for ($taskAttempt = 0; $taskAttempt -lt 10; $taskAttempt++) {
        Start-Sleep -Milliseconds 300
        try {
            $taskResponse = Invoke-WebRequest -Uri $taskUrl -UseBasicParsing -TimeoutSec 2
            if ($taskResponse.StatusCode -eq 200) { $taskRunning = $true; break }
        } catch { }
    }
}
if (-not $taskRunning) { throw 'Could not start the local review server on port 8767.' }
Start-Process $taskUrl
