param(
    [string]$Server = 'whj@10.16.81.144',
    [int]$Port = 22,
    [string]$Destination = 'D:\PythonProjects\checkpoints'
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw 'Windows OpenSSH Client (scp) is required.'
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$remoteDirectory = '/home/whj/reproject/mact_backup_20261031'
$archives = @(
    @{
        Name = 'mact-main-base-twitter2015-seed26-20261031.tar'
        SHA256 = 'd5fc180d0f5dd60aaf74cc8c315b8624d4e5e316b4ffc74a53656204ccac0083'
    },
    @{
        Name = 'mact-main-base-twitter2017-seed44-20261031.tar'
        SHA256 = '3d8441fff63054a28011896c106513662eef3418ce90dd73715832e29cd78f2b'
    }
)
foreach ($archive in $archives) {
    $target = Join-Path $Destination $archive.Name
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ieq $archive.SHA256) {
            Write-Host "Already verified: $target"
            continue
        }
        throw "An existing file has a different checksum; preserved without overwriting: $target"
    }
    $partial = "$target.$([guid]::NewGuid().ToString('N')).partial"
    $source = "${Server}:$remoteDirectory/$($archive.Name)"
    & scp -P $Port $source $partial
    if ($LASTEXITCODE -ne 0) { throw "Download failed. Partial file retained: $partial" }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ine $archive.SHA256) {
        throw "Checksum mismatch. Download retained for inspection: $partial"
    }
    Move-Item -LiteralPath $partial -Destination $target
    Write-Host "Verified: $target"
}
$archives | ForEach-Object { "$($_.SHA256)  $($_.Name)" } |
    Set-Content -LiteralPath (Join-Path $Destination 'mact-main-base-SHA256SUMS.txt') -Encoding ascii
Write-Host "Both weight archives are downloaded and verified in $Destination"
Write-Host 'Each archive contains MACT + target-classifier weights and their tokenizers.'
Write-Host 'Extract from a restored mact repository root to preserve the original relative paths.'
