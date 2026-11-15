param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $Url,

    [string] $OutputDirectory = "."
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
    throw "yt-dlp was not found on PATH."
}

$durationOutput = & yt-dlp --no-warnings --no-playlist --print duration $Url
if ($LASTEXITCODE -ne 0) {
    throw "yt-dlp failed to read the video duration."
}

$durationText = [string]($durationOutput | Select-Object -Last 1)
[double]$durationSeconds = 0
$parsed = [double]::TryParse(
    $durationText.Trim(),
    [System.Globalization.NumberStyles]::Float,
    [System.Globalization.CultureInfo]::InvariantCulture,
    [ref]$durationSeconds
)
if (-not $parsed -or $durationSeconds -lt 5) {
    throw "yt-dlp returned an invalid duration; the video must be at least five seconds long."
}

$duration = [int][Math]::Floor($durationSeconds)
$chunkSize = [int][Math]::Floor($duration / 5)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

for ($i = 0; $i -lt 5; $i++) {
    $start = $i * $chunkSize
    if ($i -eq 4) {
        $end = $duration
    }
    else {
        $end = ($i + 1) * $chunkSize
    }

    $section = "*$start-$end"
    $outputTemplate = Join-Path $OutputDirectory ("chunk_{0}.%(ext)s" -f $i)
    Write-Host "Downloading chunk $($i + 1)/5 ($start-$end seconds)..."

    & yt-dlp --no-warnings --no-playlist -f "bv*+ba/b" --merge-output-format mp4 --remux-video mp4 --download-sections $section -o $outputTemplate $Url
    if ($LASTEXITCODE -ne 0) {
        throw "yt-dlp failed while downloading chunk $($i + 1)."
    }
}
