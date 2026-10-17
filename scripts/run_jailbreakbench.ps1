param(
    [ValidateSet('harmful', 'benign')]
    [string]$Split = 'harmful',
    [ValidateRange(1, 100)]
    [int]$Limit = 2,
    [string]$CampaignId = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python data/import_jailbreakbench.py --smoke_count $Limit
    if (-not $CampaignId) {
        $CampaignId = "jbb_${Split}_smoke"
    }
    $seedPath = "data/jailbreakbench/${Split}_smoke.jsonl"
    python -m src.eval.run_campaign --config configs/smoke_test.yaml --seeds $seedPath --campaign_id $CampaignId
}
finally {
    Pop-Location
}
