<#
.SYNOPSIS
    Installs the MSI on this machine, exercises the service, and uninstalls it.

.DESCRIPTION
    CI runs this on a Windows runner, which has no GPU, so it points the service
    at the software rasterizer (Adapter=software) before installing. That covers
    the packaging, the service protocol and the CLI; it says nothing about the
    idle-eviction workaround itself, which only a real box can show.

    Needs an elevated shell. Leaves the machine as it found it.
#>
param(
    [Parameter(Mandatory = $true)][string]$Msi,
    # A second package built from the same sources: same version, own ProductCode.
    [Parameter(Mandatory = $true)][string]$UpgradeMsi,
    [string]$LogDir = $env:RUNNER_TEMP
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# Native exit codes are assertions here, not errors to be thrown on.
$PSNativeCommandUseErrorActionPreference = $false

if (-not $LogDir) { $LogDir = $env:TEMP }
$exe = Join-Path $env:ProgramFiles 'Pipette\GPU Keepalive\gpu-keepalive.exe'
$installDir = Split-Path $exe -Parent
$serviceName = 'PipetteGpuKeepalive'
$configKey = 'HKLM:\SOFTWARE\Pipette\GpuKeepalive'
$serviceLog = Join-Path $env:ProgramData 'Pipette\gpu-keepalive\gpu-keepalive.log'

function Assert([bool]$condition, [string]$message) {
    if (-not $condition) { throw "smoke test: $message" }
}

# Start-Process rather than the call operator: it gives the exit code and both
# streams the same way on every PowerShell version.
function Invoke-Gk([string[]]$Arguments) {
    $out = Join-Path $LogDir 'gk-stdout.txt'
    $err = Join-Path $LogDir 'gk-stderr.txt'
    $process = Start-Process -FilePath $exe -ArgumentList $Arguments -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $text = (Get-Content -Raw $out) + (Get-Content -Raw $err)
    Write-Host "  gpu-keepalive $($Arguments -join ' ') -> $($process.ExitCode): $text"
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Text = $text }
}

function Get-ServiceState {
    $service = Get-Service $serviceName -ErrorAction SilentlyContinue
    if ($null -eq $service) { return 'absent' }
    return $service.Status.ToString()
}

function Get-ServiceLog {
    if (Test-Path $serviceLog) { return Get-Content -Raw $serviceLog }
    return ''
}

function Invoke-Msi([string]$Package, [string]$Mode, [string]$Label) {
    $log = Join-Path $LogDir "msi-$Label.log"
    $process = Start-Process msiexec -Wait -PassThru -ArgumentList "$Mode `"$Package`" /qn /l*v `"$log`""
    Assert ($process.ExitCode -eq 0) "msiexec $Mode ($Label) exited $($process.ExitCode); see $log"
}

function Get-MachinePath {
    return [Environment]::GetEnvironmentVariable('Path', 'Machine').Split(';') | ForEach-Object { $_.TrimEnd('\') }
}

try {
    Write-Host '== install'
    New-Item -Path $configKey -Force | Out-Null
    Set-ItemProperty -Path $configKey -Name Adapter -Value 'software'
    Invoke-Msi $Msi '/i' 'install'

    Assert ((Get-ServiceState) -eq 'Running') "service state is $(Get-ServiceState)"
    # READY is logged only after the first GPU write is read back and verified.
    Assert ((Get-ServiceLog) -match 'READY adapter=\d+ .*selection=software') "service log:`n$(Get-ServiceLog)"

    $configured = (sc.exe qc $serviceName) -join "`n"
    Assert ($configured -match 'AUTO_START\s+\(DELAYED\)') "sc qc reported:`n$configured"
    $failure = (sc.exe qfailure $serviceName) -join "`n"
    Assert ($failure -match 'RESET_PERIOD \(in seconds\)\s*:\s*86400') "sc qfailure reported:`n$failure"
    Assert ($failure -match 'RESTART -- Delay = 60000') "sc qfailure reported:`n$failure"
    $flag = (sc.exe qfailureflag $serviceName) -join "`n"
    Assert ($flag -match 'TRUE') "sc qfailureflag reported:`n$flag"
    Assert ((Get-MachinePath) -contains $installDir) 'the install directory is not on PATH'

    Write-Host '== the CLI reports what SCM reports'
    $status = Invoke-Gk @('status')
    Assert ($status.ExitCode -eq 0) "status exited $($status.ExitCode)"
    Assert ($status.Text -match 'service:\s+running') "status printed:`n$($status.Text)"
    Assert ($status.Text -match 'config:\s+enabled, interval 100 ms, adapter software') "status printed:`n$($status.Text)"

    Write-Host '== foreground commands'
    $run = Invoke-Gk @('run', '--adapter', 'software', '--duration-secs', '2')
    Assert ($run.ExitCode -eq 0) "run exited $($run.ExitCode)"
    Assert ($run.Text -match 'DONE pulses=\d+ .*verified_counter=\d+') 'run printed no verified DONE line'
    Assert ((Invoke-Gk @('list')).ExitCode -eq 0) 'list failed'
    Assert ((Invoke-Gk @('--version')).ExitCode -eq 0) 'version failed'

    Write-Host '== disable, and an upgrade that must not restart it'
    Assert ((Invoke-Gk @('disable')).ExitCode -eq 0) 'disable failed'
    Assert ((Get-ServiceState) -eq 'Stopped') "service state after disable is $(Get-ServiceState)"
    Assert ((Get-ItemProperty $configKey).Enabled -eq 0) 'the configuration is still enabled'
    Assert ((Get-ServiceLog) -match 'DONE pulses=\d+ verified_counter=\d+') 'the service logged no verified DONE'
    Assert ((Invoke-Gk @('start')).ExitCode -ne 0) 'start succeeded while disabled'

    Write-Host '== disabled startup ignores invalid pulse settings'
    foreach ($setting in @('IntervalMs', 'Adapter')) {
        if ($setting -eq 'IntervalMs') {
            New-ItemProperty -Path $configKey -Name IntervalMs -PropertyType DWord -Value 0 -Force | Out-Null
        } else {
            Set-ItemProperty -Path $configKey -Name Adapter -Value 'invalid-adapter'
        }
        $start = Invoke-Gk @('start')
        Assert ($start.ExitCode -ne 0 -and $start.Text -match 'Disabled in the configuration') `
            "start did not honor disabled policy with invalid $setting"
        # Status must still expose malformed settings for the operator to fix.
        Assert ((Invoke-Gk @('status')).Text -match 'config:\s+invalid:') "status hid invalid $setting"

        # Bypass the CLI's policy check, as a boot or SCM recovery start does.
        # StartService may report failure when the disabled service exits immediately;
        # the service's log and final exit code are what determine success here.
        $logLength = (Get-ServiceLog).Length
        sc.exe start $serviceName | Out-Host
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        do {
            Start-Sleep -Milliseconds 200
            $newLog = (Get-ServiceLog).Substring($logLength)
        } while (($newLog.Length -eq 0 -or (Get-ServiceState) -ne 'Stopped') -and [DateTime]::UtcNow -lt $deadline)
        Assert ((Get-ServiceState) -eq 'Stopped') "disabled service did not stop with invalid $setting"
        Assert ($newLog -match 'disabled by configuration' -and $newLog -notmatch 'ERROR|READY') `
            "disabled startup failed with invalid ${setting}: $newLog"
        $service = Get-CimInstance Win32_Service -Filter "Name='$serviceName'"
        Assert ($service.ExitCode -eq 0 -and $service.ServiceSpecificExitCode -eq 0) `
            "disabled startup reported failure with invalid $setting"

        if ($setting -eq 'IntervalMs') {
            Remove-ItemProperty -Path $configKey -Name IntervalMs
        } else {
            Set-ItemProperty -Path $configKey -Name Adapter -Value 'software'
        }
    }

    Invoke-Msi $UpgradeMsi '/i' 'upgrade-disabled'
    Assert ((Get-ServiceState) -eq 'Stopped') 'the upgrade started a disabled service'

    Write-Host '== enable, stop, start'
    Assert ((Invoke-Gk @('enable')).ExitCode -eq 0) 'enable failed'
    Assert ((Get-ServiceState) -eq 'Running') 'not running after enable'
    Assert ((Invoke-Gk @('stop')).ExitCode -eq 0) 'stop failed'
    Assert ((Get-ServiceState) -eq 'Stopped') 'not stopped after stop'
    Assert ((Invoke-Gk @('start')).ExitCode -eq 0) 'start failed'
    Assert ((Get-ServiceState) -eq 'Running') 'not running after start'

    Write-Host '== SCM recovers from a reported configuration failure'
    Assert ((Invoke-Gk @('stop')).ExitCode -eq 0) 'stop before recovery test failed'
    New-ItemProperty -Path $configKey -Name IntervalMs -PropertyType DWord -Value 0 -Force | Out-Null
    $logLength = (Get-ServiceLog).Length
    Assert ((Invoke-Gk @('start')).ExitCode -ne 0) 'start accepted an invalid interval'
    Assert ((Get-ServiceState) -eq 'Stopped') 'failed service did not stop'
    $service = Get-CimInstance Win32_Service -Filter "Name='$serviceName'"
    Assert ($service.ExitCode -eq 1066 -and $service.ServiceSpecificExitCode -eq 1) `
        'configuration failure did not report a service-specific error to SCM'
    Assert ((Get-ServiceLog).Substring($logLength) -match 'ERROR .*IntervalMs') `
        'configuration failure was not logged'

    # Fix the cause, then leave starting entirely to SCM's configured recovery.
    # This exercises FailureActionsWhen for a reported error, without a crash.
    Remove-ItemProperty -Path $configKey -Name IntervalMs
    $logLength = (Get-ServiceLog).Length
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    do {
        Start-Sleep -Milliseconds 500
    } while ((Get-ServiceState) -ne 'Running' -and [DateTime]::UtcNow -lt $deadline)
    Assert ((Get-ServiceState) -eq 'Running') 'SCM did not restart the service after failure'
    Assert ((Get-ServiceLog).Substring($logLength) -match 'READY adapter=\d+ .*selection=software') `
        'the recovered service did not verify its first GPU pulse'

    Write-Host '== upgrade keeps an enabled service running'
    Invoke-Msi $Msi '/i' 'upgrade-enabled'
    Assert ((Get-ServiceState) -eq 'Running') "service state after upgrade is $(Get-ServiceState)"

    Write-Host '== uninstall'
    Invoke-Msi $Msi '/x' 'uninstall'
    Assert (-not (Test-Path $exe)) 'the executable survived uninstall'
    Assert ((Get-ServiceState) -eq 'absent') 'the service survived uninstall'
    Assert (-not ((Get-MachinePath) -contains $installDir)) 'the PATH entry survived uninstall'
    # Configuration and the log are operator state: uninstall leaves them behind.
    Assert (Test-Path $configKey) 'uninstall removed the configuration'
    Assert (Test-Path $serviceLog) 'uninstall removed the log'

    Write-Host 'smoke test passed'
} finally {
    Copy-Item $serviceLog (Join-Path $LogDir 'service-log.txt') -ErrorAction SilentlyContinue
    Write-Host '== service log'
    Write-Host (Get-ServiceLog)
    Remove-Item $configKey -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Split-Path $serviceLog -Parent) -Recurse -Force -ErrorAction SilentlyContinue
}
