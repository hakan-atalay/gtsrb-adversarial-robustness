$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Run([string[]]$cmd) {
    & python @cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed: python $($cmd -join ' ')" }
}
Run @("-m", "signguard.data", "--source", "torchvision")
Run @("-m", "signguard.train", "--mode", "standard", "--name", "standard", "--epochs", "30")
Run @("-m", "signguard.train", "--mode", "fast", "--name", "fast_at", "--epochs", "15")
Run @("-m", "signguard.train", "--mode", "pgd", "--name", "pgd_at", "--epochs", "15", "--steps", "3", "--alpha", "3")
Run @("-m", "signguard.train", "--mode", "trades", "--name", "trades", "--epochs", "15", "--steps", "3", "--alpha", "3")
Run @("-m", "signguard.train", "--mode", "standard", "--arch", "resnet18", "--name", "resnet18_surrogate", "--epochs", "10")
Run @("-m", "signguard.evaluate", "--models", "standard", "fast_at", "pgd_at", "trades", "resnet18_surrogate", "--n-samples", "3000")
Run @("-m", "signguard.targeted", "--models", "standard", "fast_at", "pgd_at", "trades")
Run @("-m", "signguard.transfer", "--models", "resnet18_surrogate", "standard", "fast_at", "pgd_at", "trades")
Run @("-m", "signguard.gradcam", "--models", "standard", "fast_at", "pgd_at", "trades")
Get-Content results/summary.md
