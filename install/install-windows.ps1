<#
.SYNOPSIS
  MELETRA (Windows x64): find the NEWEST release on GitHub, download it, check its SHA-256, install it.

.DESCRIPTION
  powershell -ExecutionPolicy Bypass -File install-windows.ps1                 download + run the installer (Windows asks for administrator approval)
  powershell -ExecutionPolicy Bypass -File install-windows.ps1 -Silent         the same without the wizard pages
  powershell -ExecutionPolicy Bypass -File install-windows.ps1 -Portable       unzip the portable version into -Folder\App instead (no installation)
  powershell -ExecutionPolicy Bypass -File install-windows.ps1 -DownloadOnly   only download + check

  The repository is private: the script needs access to it - either `gh auth login` (GitHub CLI) was done, or the environment variable
  GH_TOKEN holds a token that can read the repository. If the repository is made public neither is needed.
  Downloads continue where they stopped after a broken connection. The speech model (Whisper Large-v3, about 3 GB) is NOT part of the installer: the
  program uses the one already on the PC or downloads it once, with a checksum, the first time it starts.
#>
param(
    [switch]$Portable,
    [switch]$DownloadOnly,
    [switch]$Silent,
    [string]$Folder = (Join-Path $env:USERPROFILE 'Downloads\MELETRA'),
    [string]$Repo = 'twonezerotwo-hue/whispermedical',
    [string]$Tag = ''
)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Say($m) { Write-Host "[install] $m" }

$token = $env:GH_TOKEN
if (-not $token -and (Get-Command gh -ErrorAction SilentlyContinue)) { $token = (& gh auth token 2>$null) }
$headers = @{ 'User-Agent' = 'mlt-install'; 'Accept' = 'application/vnd.github+json' }
if ($token) { $headers['Authorization'] = "Bearer $token" }

try {
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=30" -Headers $headers
} catch {
    throw "Could not read the releases of $Repo ($($_.Exception.Message)). The repository is private: run 'gh auth login' once, or set GH_TOKEN to a token that can read it."
}
$release = $releases | Where-Object { -not $_.draft -and $_.tag_name -like 'windows-v*' -and ((-not $Tag) -or $_.tag_name -eq $Tag) } | Select-Object -First 1
if (-not $release) { throw "No Windows release (tag windows-v*) was found in $Repo." }
Say "release $($release.tag_name): $($release.name)"

New-Item -ItemType Directory -Force $Folder | Out-Null
$wanted = if ($Portable) { 'MELETRA-Portable-Windows-x64.zip' } else { 'MELETRA-Setup-Windows-x64.exe' }

function Get-Asset($name) {
    $asset = $release.assets | Where-Object { $_.name -eq $name } | Select-Object -First 1
    if (-not $asset) { throw "The release has no file named $name." }
    $target = Join-Path $Folder $name
    Say ("downloading {0} ({1:N0} MB) ..." -f $name, ($asset.size / 1MB))
    $curlArgs = @('--fail', '--location', '--silent', '--show-error', '--continue-at', '-', '--retry', '40', '--retry-all-errors', '--retry-delay', '5',
              '--header', 'Accept: application/octet-stream', '--user-agent', 'mlt-install')
    if ($token) { $curlArgs += @('--header', "Authorization: Bearer $token") }
    & curl.exe @curlArgs --output $target "https://api.github.com/repos/$Repo/releases/assets/$($asset.id)"
    if ($LASTEXITCODE -ne 0) { throw "Download of $name failed (curl exit code $LASTEXITCODE)." }
    if ((Get-Item $target).Length -ne $asset.size) { throw "$name has the wrong size after the download." }
    return $target
}

# a big file may be stored in small parts (NAME.part001 ...): they are downloaded, joined, and the whole file is checked below
function Get-Whole($name) {
    if ($release.assets | Where-Object { $_.name -eq $name }) { return Get-Asset $name }
    $parts = @($release.assets | Where-Object { $_.name -like "$name.part*" } | Sort-Object name)
    if ($parts.Count -eq 0) { throw "The release has neither $name nor its parts." }
    $partFiles = @(foreach ($part in $parts) { Get-Asset $part.name })
    $target = Join-Path $Folder $name
    Say "joining $($parts.Count) parts ..."
    $out = [IO.File]::Create($target)
    try {
        foreach ($f in $partFiles) { $in = [IO.File]::OpenRead($f); try { $in.CopyTo($out) } finally { $in.Dispose() } }
    } finally { $out.Dispose() }
    $partFiles | Remove-Item -Force
    return $target
}

$file = Get-Whole $wanted
$sums = Get-Asset 'SHA256SUMS.txt'
$expected = (Get-Content $sums | Where-Object { $_ -like "*$wanted" } | Select-Object -First 1).Split(' ')[0].ToLower()
$actual = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
if (-not $expected -or $actual -ne $expected) { Remove-Item $file -Force; throw "SHA-256 of $wanted does not match the release ($actual). The file was deleted; run the script again." }
Say "SHA-256 verified: $actual"

if ($DownloadOnly) { Say "downloaded to $file"; return }
if ($Portable) {
    $app = Join-Path $Folder 'App'
    Say "unzipping into $app"
    Expand-Archive -LiteralPath $file -DestinationPath $app -Force
    Say "done. Start $app\MELETRA\MELETRA.exe"
    return
}
$arguments = if ($Silent) { @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART') } else { @() }
Say 'starting the installer (Windows will ask for administrator approval; a running copy of the program is closed first) ...'
$process = Start-Process -FilePath $file -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "The installer ended with exit code $($process.ExitCode)." }
Say 'installed. Start "MELETRA" from the Start menu.'
