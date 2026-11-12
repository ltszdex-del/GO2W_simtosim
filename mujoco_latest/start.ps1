$taskRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$mujocoPython = (Get-Command python -ErrorAction Stop).Source
& $mujocoPython (Join-Path $taskRoot 'run.py') @args
