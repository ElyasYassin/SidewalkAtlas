# Wait until scene generation and every render shard have exited, then build
# the dataset tables. Runs detached; safe to re-run.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$log = "D:/sidewalk_crack_synth/finish.log"
"=== waiting $(Get-Date)" | Out-File -Append -Encoding utf8 $log
do {
    Start-Sleep 60
    $busy = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like '*render_crack_dataset.py*' -or $_.CommandLine -like '*generate_crack_scenes.py*' }
} while ($busy)
"=== building $(Get-Date)" | Out-File -Append -Encoding utf8 $log
python scripts/crack_data/build_crack_dataset.py 2>&1 | Out-File -Append -Encoding utf8 $log
python scripts/crack_data/contact_sheet.py --render D:/sidewalk_crack_synth/render --n 24 --cols 4 --out D:/sidewalk_crack_synth/final_sheet.jpg 2>&1 | Out-File -Append -Encoding utf8 $log
"=== done $(Get-Date)" | Out-File -Append -Encoding utf8 $log
