# While the crack renderer runs, delete Omniverse texture-cache entries for
# the scene textures (albedo/normal/roughness). Each scene's textures are
# used once, and the cache lives on C: -- 500 scenes is ~35 GB, which filled
# the drive. Only this run's entries are touched; older cache is left alone.
$t = "$env:LOCALAPPDATA\ov\cache\texturecache"
$since = Get-Date '2026-10-13 19:00'
do {
    Get-ChildItem $t -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $since -and
                       $_.LastWriteTime -lt (Get-Date).AddMinutes(-5) -and
                       $_.Name -match '^(albedo|normal|roughness|_validate_labels)_' } |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Start-Sleep 120
    $busy = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*render_crack_dataset.py*' }
} while ($busy)
