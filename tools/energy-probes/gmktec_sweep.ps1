# Token sweep on a GMKtec box: each length at -r 2 and -r 6, sampling ryzenadj
# PPT VALUE FAST alongside. Refuses to start while the worker has an inference child.
$ErrorActionPreference = "Continue"
$ra = "C:\tools\ryzenadj\ryzenadj.exe"
$LB = "C:\pipette\.pipette\runtimes\llama-cpp__ggml-org_llama.cpp__b10516__windows-x64-vulkan\blobs\llama-bench.exe"
$M  = "C:\pipette\.pipette\models\LiquidAI__LFM2.5-1.2B-I_1eb107bb\blobs\LFM2.5-1.2B-Instruct-Q4_0.gguf"
$kids = Get-CimInstance Win32_Process | Where-Object { $_.Name -match "llama|pipette" -and $_.Name -ne "pipette.exe" }
if ($kids) { "BUSY: " + ($kids.Name -join ","); exit }
Get-Process pipette -ErrorAction SilentlyContinue | ForEach-Object { "pipette worker pid $($_.Id) still up (mediator stopped?)" }
$samp = "C:\pipette\energy_samples.txt"; Remove-Item $samp -ErrorAction SilentlyContinue
$job = Start-Job -ScriptBlock { param($ra,$f) while ($true) { $o = & $ra --info 2>&1 | Select-String "PPT VALUE FAST"; $v = (($o -split "\|")[2]).Trim(); Add-Content $f ("{0:F3} {1}" -f ((Get-Date).ToUniversalTime() - [datetime]'1970-01-01').TotalSeconds, $v) } } -ArgumentList $ra,$samp
Start-Sleep 3
function run($phase,$n,$r) {
  if ($phase -eq "decode") { $a = @("--n-prompt","0","--n-gen","$n","--n-depth","512") } else { $a = @("--n-prompt","$n","--n-gen","0") }
  $t0 = ((Get-Date).ToUniversalTime() - [datetime]'1970-01-01').TotalSeconds
  $out = & $LB --model $M --mmap 0 -r $r @a --output json 2>$null | Out-String
  $t1 = ((Get-Date).ToUniversalTime() - [datetime]'1970-01-01').TotalSeconds
  $j = $out.Substring($out.IndexOf('['), $out.LastIndexOf(']') - $out.IndexOf('[') + 1) | ConvertFrom-Json
  $ts = ($j[0].samples_ns | Measure-Object -Sum).Sum / 1e9
  "$phase n=$n r=$r bench_measured_s=$('{0:F3}' -f $ts) t0=$('{0:F3}' -f $t0) t1=$('{0:F3}' -f $t1) elapsed_s=$('{0:F3}' -f ($t1-$t0))"
}
run decode 50 2 | Out-Null   # warm
foreach ($n in 50,100,200,400,800) { run decode $n 2; run decode $n 6 }
foreach ($n in 256,512,1024,2048,4096) { run prefill $n 2; run prefill $n 6 }
run decode 800 14
run prefill 4096 14
Stop-Job $job; Remove-Job $job
"SAMPLES_BEGIN"; Get-Content $samp; "SAMPLES_END"
