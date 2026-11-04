# Samples ryzenadj socket power idle, under a 12 s 16-thread CPU load, and after.
# Refuses to run while the pipette worker has an inference child.
$ra = "C:\tools\ryzenadj\ryzenadj.exe"
function snap($tag) {
  $o = & $ra --info 2>&1 | Select-String "PPT VALUE (FAST|SLOW)|STAPM VALUE|THM VALUE CORE"
  "$tag " + (($o | ForEach-Object { ($_ -replace "\s+\|\s+", " ").Trim("| ") }) -join " ; ")
}
$kids = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq (Get-Process pipette).Id }
if ($kids) { "BUSY: " + ($kids.Name -join ","); exit }
snap "idle"
$jobs = 1..16 | ForEach-Object { Start-Job { $t = [Diagnostics.Stopwatch]::StartNew(); while ($t.ElapsedMilliseconds -lt 12000) { [math]::Sqrt(12345) | Out-Null } } }
Start-Sleep 5; snap "load+5s"; Start-Sleep 4; snap "load+9s"
$jobs | Wait-Job | Remove-Job
Start-Sleep 3; snap "after+3s"
