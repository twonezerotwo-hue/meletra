# Upgrade scenarios of build_tools/installer.iss (per user, silent) with tiny stand-in programs; driven by scripts/installer_upgrade_check.py.
#   1. a replica of the 4.0.0 setup is installed into a folder called "Medical Lecture Transcriber"
#   2. the MELETRA setup is installed over it: one program is left (MELETRA, in a folder of the new name), the old folder, program file and Start menu folder are gone
#   3. a running copy of the program stops the upgrade and no file is touched
#   4. the uninstaller leaves the user's data folder (%LOCALAPPDATA%\MedicalLectureTranscriber) exactly as it was (it is only listed, never written)
param([Parameter(Mandatory)][string]$Work, [Parameter(Mandatory)][string]$Out)
$ErrorActionPreference = 'Continue'
$t = $Work
$key = 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{6F0B7C52-3A1E-4E7D-9C54-8D2A7B1E4F60}_is1'
$data = "$env:LOCALAPPDATA\MedicalLectureTranscriber"
function Snap { Get-ChildItem $data -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\models\\|\\temp\\|\\cache\\|\\logs\\' } | ForEach-Object { "$($_.FullName.Substring($data.Length)) $($_.Length) $($_.LastWriteTimeUtc.Ticks)" } | Sort-Object }
$lines = New-Object System.Collections.Generic.List[string]
function Say($m) { $lines.Add($m); Write-Host $m }

Say "--- before"
if (Test-Path $key) { Say "an HKCU install already exists - stopping"; $lines | Set-Content $Out; exit 2 }
$before = Snap
Say "real data snapshot: $($before.Count) files"
$instRoot = "$t\inst"
Remove-Item -Recurse -Force $instRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $instRoot | Out-Null

Say "--- 1. install the replica of the 4.0.0 setup (per user) into a folder called 'Medical Lecture Transcriber'"
$old = "$instRoot\Medical Lecture Transcriber"
$p = Start-Process "$t\out\old-setup.exe" -ArgumentList '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/DIR=`"$old`"", "/LOG=$t\old-install.log" -Wait -PassThru
Say "exit $($p.ExitCode)"
$i = Get-ItemProperty $key
Say "registry: $($i.DisplayName) $($i.DisplayVersion) at $($i.InstallLocation)"
Say "old exe present: $(Test-Path "$old\MedicalLectureTranscriber.exe")"
$menuOld = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Medical Lecture Transcriber"
Say "old start menu folder present: $(Test-Path $menuOld)"

Say "--- 2. install the new MELETRA setup over it (silent, no /DIR: the upgrade has to find the right place itself)"
$p = Start-Process "$t\out\new-setup.exe" -ArgumentList '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/LOG=$t\new-install.log" -Wait -PassThru
Say "exit $($p.ExitCode)"
$i = Get-ItemProperty $key
Say "registry: $($i.DisplayName) $($i.DisplayVersion) at $($i.InstallLocation)"
$new = $i.InstallLocation.TrimEnd('\')
Say "new exe present: $(Test-Path "$new\MELETRA.exe")  (folder name: $(Split-Path $new -Leaf))"
Say "old folder gone: $(-not (Test-Path $old))"
Say "old exe anywhere: $(Test-Path "$new\MedicalLectureTranscriber.exe")"
Say "new payload file: $(Test-Path "$new\sub\new.txt")  old payload file: $(Test-Path "$new\sub\old.txt")"
Say "old start menu folder gone: $(-not (Test-Path $menuOld))"
$uninst = Get-ChildItem "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" | Get-ItemProperty | Where-Object { $_.DisplayName -match 'MELETRA|WhisperMedical|Medical Lecture Transcriber' }
Say "programs listed for this user: $(@($uninst).Count) ($(($uninst | ForEach-Object { $_.DisplayName }) -join ', '))"

Say "--- 3. a running copy of the program blocks the upgrade WITHOUT touching files"
$proc = Start-Process "$new\MELETRA.exe" -ArgumentList '--wait' -PassThru -WindowStyle Hidden
Start-Sleep 2
$p = Start-Process "$t\out\new-setup.exe" -ArgumentList '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/LOG=$t\new-install-running.log" -Wait -PassThru
Say "exit with the program running: $($p.ExitCode) (non-zero expected)"
Say "files intact: $(Test-Path "$new\MELETRA.exe")"
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 1

Say "--- 4. uninstall (silent): data folder untouched"
$un = Get-ChildItem $new -Filter 'unins*.exe' | Select-Object -First 1
$p = Start-Process $un.FullName -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -Wait -PassThru
Start-Sleep 3
Say "exit $($p.ExitCode); registry key gone: $(-not (Test-Path $key)); folder gone: $(-not (Test-Path $new))"

$after = Snap
Say "--- real data folder unchanged: $((@(Compare-Object $before $after)).Count -eq 0)  ($($after.Count) files)"
$lines | Set-Content $Out

