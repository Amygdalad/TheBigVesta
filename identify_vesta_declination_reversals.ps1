# Requires: PowerShell 5+
param(
    [string]$InFile = "vesta_ephemeris_weekly.txt",
    [string]$Output = "vesta_declination_reversals.csv",
    [switch]$Preview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (!(Test-Path -LiteralPath $InFile)) {
    throw "Input file not found: $InFile"
}

$inTable = $false
$dates = New-Object System.Collections.Generic.List[datetime]
$decs  = New-Object System.Collections.Generic.List[double]

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$styles = [System.Globalization.DateTimeStyles]::AssumeUniversal

function Parse-DecToDeg([string]$decStr) {
    $parts = $decStr.Trim() -split '\s+'
    if ($parts.Count -ne 3) { throw "Unexpected Dec format: $decStr" }
    $sign = if ($parts[0].StartsWith('-')) { -1.0 } else { 1.0 }
    $deg = [double]($parts[0].Replace('+','').Replace('-',''))
    $arcmin = [double]$parts[1]
    $arcsec = [double]$parts[2]
    return $sign * ($deg + $arcmin/60.0 + $arcsec/3600.0)
}

Get-Content -LiteralPath $InFile | ForEach-Object {
    $line = $_
    $trim = $line.Trim()
    if ($trim -eq '$$SOE') { $inTable = $true; return }
    if ($trim -eq '$$EOE') { $inTable = $false; break }
    if (-not $inTable) { return }
    if (-not $trim) { return }

    $parts = $line.Split(',') | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 6) { return }

    $dtStr = $parts[0]
    $decStr = $parts[4]
    try {
        $dt = [datetime]::ParseExact($dtStr, 'yyyy-MMM-dd HH:mm', $culture, $styles)
    } catch {
        return
    }
    try {
        $decDeg = Parse-DecToDeg $decStr
    } catch {
        return
    }
    [void]$dates.Add($dt)
    [void]$decs.Add($decDeg)
}

if ($dates.Count -lt 3) {
    throw "Parsed only $($dates.Count) rows; need at least 3."
}

# Compute typical spacing (median seconds)
$spacings = New-Object System.Collections.Generic.List[double]
for ($i=0; $i -lt $dates.Count-1; $i++) {
    $dtSec = ($dates[$i+1] - $dates[$i]).TotalSeconds
    [void]$spacings.Add([double]$dtSec)
}
if ($spacings.Count -eq 0) { throw 'No spacings computed.' }
$sorted = $spacings.ToArray(); [Array]::Sort($sorted)
$median = $sorted[[int][math]::Floor($sorted.Length/2)]
$hSec = [double]$median

$results = New-Object System.Collections.Generic.List[pscustomobject]

for ($i=1; $i -lt $decs.Count-1; $i++) {
    $y0 = [double]$decs[$i-1]
    $y1 = [double]$decs[$i]
    $y2 = [double]$decs[$i+1]
    $d0 = $y1 - $y0
    $d1 = $y2 - $y1

    if ($d0 -eq 0 -or $d1 -eq 0) { continue }

    $kind = $null
    if ($d0 -gt 0 -and $d1 -lt 0) { $kind = 'declination_max' }
    elseif ($d0 -lt 0 -and $d1 -gt 0) { $kind = 'declination_min' }
    else { continue }

    $denom = ($y0 - 2.0*$y1 + $y2)
    if ([math]::Abs($denom) -lt 1e-12) {
        $tExt = $dates[$i]
        $yTau = $y1
    } else {
        $tOffset = ($hSec * ($y0 - $y2)) / (2.0 * $denom)
        if ($tOffset -gt $hSec) { $tOffset = $hSec }
        if ($tOffset -lt -$hSec) { $tOffset = -$hSec }
        $tExt = $dates[$i].AddSeconds($tOffset)
        $tau = $tOffset / $hSec
        $yTau = ($y0 * 0.5 * $tau * ($tau - 1.0)) + ($y1 * (1.0 - $tau*$tau)) + ($y2 * 0.5 * $tau * ($tau + 1.0))
    }

    $results.Add([pscustomobject]@{
        datetime_utc     = $tExt.ToString('yyyy-MM-dd HH:mm')
        reversal_type    = $kind
        declination_deg  = [math]::Round($yTau, 6)
        row_index_center = $i
    }) | Out-Null
}

# Write CSV
$results | Export-Csv -LiteralPath $Output -NoTypeInformation -Encoding UTF8

if ($Preview) {
    $results | Select-Object -First 20 | Format-Table -AutoSize | Out-String | Write-Host
    Write-Host ("Total declination reversals detected: {0}" -f $results.Count)
} else {
    Write-Host ("Wrote {0} declination reversals to {1}" -f $results.Count, $Output)
}
