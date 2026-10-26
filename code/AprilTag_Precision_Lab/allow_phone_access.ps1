$ErrorActionPreference = 'Stop'
$resultPath = Join-Path $PSScriptRoot 'firewall_result.txt'
$python = Join-Path $env:USERPROFILE 'miniconda3\envs\lerobot\python.exe'
try {
    $rule = Get-NetFirewallRule -DisplayName 'AprilTag Precision Lab 8443' -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -DisplayName 'AprilTag Precision Lab 8443' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443 -RemoteAddress LocalSubnet -Program $python -Profile Any | Out-Null
    }
    $rule = Get-NetFirewallRule -DisplayName 'AprilTag Precision Lab 8443'
    if ($rule.Enabled -ne 'True') { throw 'Rule is not enabled' }
    'OK: TCP 8443; LocalSubnet; lerobot python only.' | Set-Content -LiteralPath $resultPath -Encoding UTF8
} catch {
    $_.Exception.Message | Set-Content -LiteralPath $resultPath -Encoding UTF8
    exit 1
}
