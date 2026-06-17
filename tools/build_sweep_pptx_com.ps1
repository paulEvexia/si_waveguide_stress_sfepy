param(
    [string]$TemplatePath = "$PSScriptRoot\..\template.pptx",
    [string]$ManifestPath = "$PSScriptRoot\..\results\sweep_ppt_manifest.json",
    [string]$OutputPath = "$PSScriptRoot\..\si_waveguide_stress_dbr_sweep_report_template_matched.pptx",
    [int]$BaseSlideIndex = 2
)

$ErrorActionPreference = "Stop"

function Add-TextBox {
    param(
        [object]$Slide,
        [string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$FontSize = 14,
        [bool]$Bold = $false
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Bold = if ($Bold) { -1 } else { 0 }
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    return $shape
}

function Add-ContainedPicture {
    param(
        [object]$Slide,
        [string]$Path,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        $missing = Add-TextBox -Slide $Slide -Text "Missing image:`n$Path" -Left $Left -Top $Top -Width $Width -Height $Height -FontSize 8
        $missing.TextFrame.TextRange.Font.Color.RGB = 255
        return $missing
    }

    Add-Type -AssemblyName System.Drawing
    $img = [System.Drawing.Image]::FromFile($Path)
    try {
        $imageRatio = $img.Width / $img.Height
        $boxRatio = $Width / $Height

        if ($imageRatio -gt $boxRatio) {
            $picWidth = $Width
            $picHeight = $Width / $imageRatio
            $picLeft = $Left
            $picTop = $Top + (($Height - $picHeight) / 2.0)
        } else {
            $picHeight = $Height
            $picWidth = $Height * $imageRatio
            $picLeft = $Left + (($Width - $picWidth) / 2.0)
            $picTop = $Top
        }
    } finally {
        $img.Dispose()
    }

    return $Slide.Shapes.AddPicture($Path, $false, $true, $picLeft, $picTop, $picWidth, $picHeight)
}

function New-TemplateBasedSlide {
    param(
        [object]$Presentation,
        [int]$BaseSlideIndex
    )

    $Presentation.Slides.Item($BaseSlideIndex).Duplicate() | Out-Null
    $slide = $Presentation.Slides.Item($BaseSlideIndex + 1)
    $slide.MoveTo($Presentation.Slides.Count)
    $slide = $Presentation.Slides.Item($Presentation.Slides.Count)

    for ($i = $slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $slide.Shapes.Item($i)
        $deleteShape = $false

        try {
            if ($shape.Type -eq 14) {
                $deleteShape = $true
            }
            if ($shape.Name -like "*Placeholder*") {
                $deleteShape = $true
            }
            if ($shape.Name -like "Title*") {
                $deleteShape = $true
            }
        } catch {
            $deleteShape = $false
        }

        if ($deleteShape) {
            $shape.Delete()
        }
    }

    return $slide
}

function Add-Panel {
    param(
        [object]$Slide,
        [string]$Label,
        [string]$ImagePath,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height
    )

    $caption = Add-TextBox -Slide $Slide -Text $Label -Left $Left -Top ($Top - 14) -Width $Width -Height 12 -FontSize 8 -Bold $true
    $caption.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    Add-ContainedPicture -Slide $Slide -Path $ImagePath -Left $Left -Top $Top -Width $Width -Height $Height | Out-Null
}

function Add-SummarySlide {
    param(
        [object]$Presentation,
        [object]$Manifest,
        [double]$SlideW,
        [double]$SlideH,
        [int]$BaseSlideIndex
    )

    $slide = New-TemplateBasedSlide -Presentation $Presentation -BaseSlideIndex $BaseSlideIndex
    Add-TextBox -Slide $slide -Text "DBR center wavelength shift summary" -Left 28 -Top 18 -Width ($SlideW - 56) -Height 30 -FontSize 20 -Bold $true | Out-Null
    Add-TextBox -Slide $slide -Text "Shift is reported in pm. Lines use intrinsic stress or reference temperature as the legend." -Left 28 -Top 46 -Width ($SlideW - 56) -Height 18 -FontSize 10 | Out-Null

    $gap = 20.0
    $left = 28.0
    $top = 78.0
    $plotW = ($SlideW - (2.0 * $left) - $gap) / 2.0
    $plotH = $SlideH - $top - 72.0

    Add-ContainedPicture -Slide $slide -Path $Manifest.summary_slide.pm_vs_temperature -Left $left -Top $top -Width $plotW -Height $plotH | Out-Null
    Add-ContainedPicture -Slide $slide -Path $Manifest.summary_slide.pm_vs_stress -Left ($left + $plotW + $gap) -Top $top -Width $plotW -Height $plotH | Out-Null
}

function Add-SweepSlide {
    param(
        [object]$Presentation,
        [object]$Entry,
        [double]$SlideW,
        [double]$SlideH,
        [int]$BaseSlideIndex
    )

    $slide = New-TemplateBasedSlide -Presentation $Presentation -BaseSlideIndex $BaseSlideIndex
    $sigma = [double]$Entry.sigma_intrinsic_si_MPa
    $tref = [double]$Entry.T_ref_C
    $pm = [double]$Entry.dbr_delta_wavelength_pm
    $title = "Tref = {0:N0} C | intrinsic Si stress = {1:N0} MPa | DBR shift = {2:N1} pm | {3}" -f $tref, $sigma, $pm, $Entry.section

    Add-TextBox -Slide $slide -Text $title -Left 28 -Top 16 -Width ($SlideW - 56) -Height 24 -FontSize 15 -Bold $true | Out-Null

    $marginX = 28.0
    $topRow = 70.0
    $gapX = 12.0
    $gapY = 36.0
    $panelW = ($SlideW - (2.0 * $marginX) - (2.0 * $gapX)) / 3.0
    $panelH = ($SlideH - $topRow - 72.0 - $gapY) / 2.0
    $bottomRow = $topRow + $panelH + $gapY

    $labels = @(
        @("sigma_xx [MPa]", $Entry.images.sigma_xx, 0, $topRow),
        @("sigma_yy [MPa]", $Entry.images.sigma_yy, 1, $topRow),
        @("sigma_xy [MPa]", $Entry.images.sigma_xy, 2, $topRow),
        @("eps_xx", $Entry.images.eps_xx, 0, $bottomRow),
        @("eps_yy", $Entry.images.eps_yy, 1, $bottomRow),
        @("eps_xy", $Entry.images.eps_xy, 2, $bottomRow)
    )

    foreach ($item in $labels) {
        $left = $marginX + ([int]$item[2] * ($panelW + $gapX))
        Add-Panel -Slide $slide -Label $item[0] -ImagePath $item[1] -Left $left -Top $item[3] -Width $panelW -Height $panelH
    }
}

if (-not (Test-Path -LiteralPath $TemplatePath)) {
    throw "Template not found: $TemplatePath"
}
if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)

$powerPoint = New-Object -ComObject PowerPoint.Application
$presentation = $null

try {
    $presentation = $powerPoint.Presentations.Open([System.IO.Path]::GetFullPath($TemplatePath), $true, $false, $false)
    $slideW = [double]$presentation.PageSetup.SlideWidth
    $slideH = [double]$presentation.PageSetup.SlideHeight
    $templateSlideCount = $presentation.Slides.Count
    if ($BaseSlideIndex -lt 1 -or $BaseSlideIndex -gt $templateSlideCount) {
        throw "BaseSlideIndex $BaseSlideIndex is outside the template slide range 1..$templateSlideCount"
    }

    Add-SummarySlide -Presentation $presentation -Manifest $manifest -SlideW $slideW -SlideH $slideH -BaseSlideIndex $BaseSlideIndex

    foreach ($entry in $manifest.slides) {
        Add-SweepSlide -Presentation $presentation -Entry $entry -SlideW $slideW -SlideH $slideH -BaseSlideIndex $BaseSlideIndex
    }

    for ($i = $templateSlideCount; $i -ge 1; $i--) {
        $presentation.Slides.Item($i).Delete()
    }

    if (Test-Path -LiteralPath $outputFullPath) {
        Remove-Item -LiteralPath $outputFullPath -Force
    }

    $presentation.SaveAs($outputFullPath)
    Write-Host "Saved PPTX: $outputFullPath"
    Write-Host "Slides: $($presentation.Slides.Count)"
} finally {
    if ($presentation -ne $null) {
        $presentation.Close()
    }
    $powerPoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint) | Out-Null
}
