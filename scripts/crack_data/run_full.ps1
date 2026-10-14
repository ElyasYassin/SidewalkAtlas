# Full synthetic crack run: scenes -> render -> dataset tables.
# Detached-friendly: every step is resumable, so re-running continues.
#   powershell -ExecutionPolicy Bypass -File scripts/crack_data/run_full.ps1
param([int]$Scenes = 500, [int]$Views = 16, [int]$Workers = 3)

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$log = "D:/sidewalk_crack_synth/run_full.log"
"=== start $(Get-Date)" | Out-File -Append -Encoding utf8 $log

python scripts/crack_data/generate_crack_scenes.py --n $Scenes --workers $Workers *>> $log
"=== scenes done $(Get-Date)" | Out-File -Append -Encoding utf8 $log

# The renderer skips scenes already in frames.jsonl; retry if Isaac exits early.
for ($i = 0; $i -lt 5; $i++) {
    & C:\isaacsim\python.bat scripts/crack_data/render_crack_dataset.py --views $Views *>> $log
    $done = (Get-Content D:/sidewalk_crack_synth/render/frames.jsonl -ErrorAction SilentlyContinue |
             ForEach-Object { ($_ | ConvertFrom-Json).scene_id } | Sort-Object -Unique).Count
    "=== render pass ${i}: ${done} scenes rendered $(Get-Date)" | Out-File -Append -Encoding utf8 $log
    if ($done -ge $Scenes) { break }
}

python scripts/crack_data/build_crack_dataset.py *>> $log
"=== done $(Get-Date)" | Out-File -Append -Encoding utf8 $log
