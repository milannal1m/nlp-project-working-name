#requires -Version 5.1

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    # Re-run completed outputs. This also passes --overwrite to every E*.py call.
    [switch]$Force
)

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Seed sweep for the cross-seed std analysis. Seed 42 is already done and lives
# in summaries/ and results/; this wrapper writes each new seed to its own
# summaries_seed<N>/ and results_seed<N>/ so nothing overwrites the seed-42 run
# and so the output filename collision (which does NOT include the seed) is
# avoided. E0.py-E5.py only accept sample sizes 1, 20, 50, 100, or 500.
$SEEDS = @(0, 1)
$TRAIN_SAMPLE = 500
$SAMPLE = 500
$VARIANTS = @("E0", "E1", "E5")
$DATASETS = @("xsum", "cnn_dailymail")
$DEVICE = "cuda"
# =============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    # Native non-zero exit codes are handled explicitly so one failed run does
    # not stop the rest of the batch.
    $PSNativeCommandUseErrorActionPreference = $false
}

$TransformerDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $TransformerDir
$LogDir = Join-Path $TransformerDir "logs_local_seeds"
$AllowedVariants = @("E0", "E1", "E2", "E3", "E4", "E5")
$AllowedDatasets = @("xsum", "cnn_dailymail")
$AllowedSampleSizes = @(1, 20, 50, 100, 500)
$script:PythonExe = $null
$script:RunResults = @()
$script:AnyFailed = $false

function Test-NonEmptyFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    try {
        return (Get-Item -LiteralPath $Path).Length -gt 0
    }
    catch {
        return $false
    }
}

function Test-AllNonEmptyFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths
    )

    foreach ($path in $Paths) {
        if (-not (Test-NonEmptyFile -Path $path)) {
            return $false
        }
    }
    return $true
}

function Format-RunDuration {
    param(
        [Parameter(Mandatory = $true)]
        [TimeSpan]$Elapsed
    )

    $totalHours = [int][Math]::Floor($Elapsed.TotalHours)
    return "{0:00}:{1:00}:{2:00}.{3:000}" -f `
        $totalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
}

function Add-RunResult {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Seed,

        [Parameter(Mandatory = $true)]
        [string]$Variant,

        [Parameter(Mandatory = $true)]
        [string]$Dataset,

        [Parameter(Mandatory = $true)]
        [string]$Task,

        [Parameter(Mandatory = $true)]
        [ValidateSet("OK", "FAILED", "SKIPPED")]
        [string]$Status,

        [Parameter(Mandatory = $true)]
        [string]$Duration,

        [Parameter(Mandatory = $true)]
        [string]$LogFile
    )

    $script:RunResults += [pscustomobject]@{
        Seed     = $Seed
        Variant  = $Variant
        Dataset  = $Dataset
        Task     = $Task
        Status   = $Status
        Duration = $Duration
        LogFile  = $LogFile
    }
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Parts
    )

    $displayParts = foreach ($part in $Parts) {
        if ($part -match '[\s"]') {
            '"' + $part.Replace('"', '\"') + '"'
        }
        else {
            $part
        }
    }
    return [string]::Join(" ", $displayParts)
}

function Add-SkippedRun {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Seed,

        [Parameter(Mandatory = $true)]
        [string]$Variant,

        [Parameter(Mandatory = $true)]
        [string]$Dataset,

        [Parameter(Mandatory = $true)]
        [string]$Task,

        [Parameter(Mandatory = $true)]
        [string]$Reason,

        [Parameter(Mandatory = $true)]
        [string]$LogFile
    )

    Write-Host "[SKIPPED] seed=$Seed $Variant / $Dataset / $Task - $Reason" -ForegroundColor Yellow
    Add-RunResult `
        -Seed $Seed `
        -Variant $Variant `
        -Dataset $Dataset `
        -Task $Task `
        -Status "SKIPPED" `
        -Duration "00:00:00.000" `
        -LogFile $LogFile
}

function Invoke-LoggedRun {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Seed,

        [Parameter(Mandatory = $true)]
        [string]$Variant,

        [Parameter(Mandatory = $true)]
        [string]$Dataset,

        [Parameter(Mandatory = $true)]
        [string]$Task,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string[]]$ExpectedFiles,

        [Parameter(Mandatory = $true)]
        [string]$LogFile,

        [string]$FailurePattern
    )

    $commandParts = @($script:PythonExe) + @($Arguments)
    $displayCommand = Format-CommandLine -Parts $commandParts

    if ($WhatIfPreference) {
        Write-Host "[WHATIF] Would run: $displayCommand" -ForegroundColor Cyan
        Add-RunResult `
            -Seed $Seed `
            -Variant $Variant `
            -Dataset $Dataset `
            -Task $Task `
            -Status "SKIPPED" `
            -Duration "00:00:00.000" `
            -LogFile $LogFile
        return "SKIPPED"
    }

    Write-Host ""
    Write-Host ("=" * 88) -ForegroundColor DarkGray
    Write-Host "[RUN] seed=$Seed $Variant / $Dataset / $Task" -ForegroundColor Cyan
    Write-Host "Command: $displayCommand"
    Write-Host "Log:     $LogFile"

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = -1
    $caughtError = $null
    $locationPushed = $false
    $previousErrorActionPreference = $ErrorActionPreference

    try {
        # Merge native stderr into stdout, save the combined stream, and keep it
        # visible. The call is synchronous; this Python process fully exits
        # before the wrapper advances to the next run.
        Push-Location -LiteralPath $RepoRoot
        $locationPushed = $true
        $ErrorActionPreference = "Continue"
        & $script:PythonExe @Arguments 2>&1 |
            Tee-Object -FilePath $LogFile |
            ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    }
    catch {
        $caughtError = $_
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($locationPushed) {
            Pop-Location
        }
        $stopwatch.Stop()
    }

    if ($null -ne $caughtError) {
        $errorLine = "Wrapper error while launching Python: $caughtError"
        $errorLine |
            Tee-Object -FilePath $LogFile |
            ForEach-Object { Write-Host $_ -ForegroundColor Red }
        $exitCode = -1
    }

    $missingOutputs = @(
        foreach ($expectedFile in $ExpectedFiles) {
            if (-not (Test-NonEmptyFile -Path $expectedFile)) {
                $expectedFile
            }
        }
    )

    $status = "OK"
    $failureReason = $null

    if ($exitCode -ne 0) {
        $status = "FAILED"
        $failureReason = "Python exited with code $exitCode"
    }
    elseif ($missingOutputs.Count -gt 0) {
        $status = "FAILED"
        $failureReason = "expected non-empty output was not produced: $($missingOutputs -join ', ')"
    }
    elseif (
        $FailurePattern -and
        (Select-String -LiteralPath $LogFile -Pattern $FailurePattern -Quiet)
    ) {
        $status = "FAILED"
        $failureReason = "the evaluator reported one or more failed files"
    }

    $duration = Format-RunDuration -Elapsed $stopwatch.Elapsed
    if ($status -eq "OK") {
        Write-Host "[OK] seed=$Seed $Variant / $Dataset / $Task completed in $duration" -ForegroundColor Green
    }
    else {
        $script:AnyFailed = $true
        Write-Host "[FAILED] seed=$Seed $Variant / $Dataset / $Task after $duration - $failureReason" -ForegroundColor Red
        Write-Host "         See $LogFile" -ForegroundColor Red
    }

    Add-RunResult `
        -Seed $Seed `
        -Variant $Variant `
        -Dataset $Dataset `
        -Task $Task `
        -Status $status `
        -Duration $duration `
        -LogFile $LogFile

    return $status
}

function Invoke-Preflight {
    $problems = @()

    Write-Host "Running preflight checks..." -ForegroundColor Cyan

    if ($AllowedSampleSizes -notcontains [int]$TRAIN_SAMPLE) {
        $problems += "TRAIN_SAMPLE=$TRAIN_SAMPLE is invalid. E0.py-E5.py accept only: $($AllowedSampleSizes -join ', ')."
    }
    if ($AllowedSampleSizes -notcontains [int]$SAMPLE) {
        $problems += "SAMPLE=$SAMPLE is invalid. E0.py-E5.py accept only: $($AllowedSampleSizes -join ', ')."
    }
    if (@($SEEDS).Count -eq 0) {
        $problems += "SEEDS cannot be empty."
    }
    foreach ($seed in @($SEEDS)) {
        if (-not ($seed -is [int]) -and -not ([int]::TryParse([string]$seed, [ref]([int]0)))) {
            $problems += "Seed '$seed' is not an integer."
        }
    }
    if (@($SEEDS | Select-Object -Unique).Count -ne @($SEEDS).Count) {
        $problems += "SEEDS contains duplicate entries."
    }
    if (@($VARIANTS).Count -eq 0) {
        $problems += "VARIANTS cannot be empty."
    }
    if (@($DATASETS).Count -eq 0) {
        $problems += "DATASETS cannot be empty."
    }
    if ([string]::IsNullOrWhiteSpace([string]$DEVICE)) {
        $problems += "DEVICE cannot be empty."
    }

    foreach ($variant in @($VARIANTS)) {
        if ($AllowedVariants -cnotcontains [string]$variant) {
            $problems += "Unsupported variant '$variant'. Allowed values: $($AllowedVariants -join ', ')."
        }
    }
    foreach ($dataset in @($DATASETS)) {
        if ($AllowedDatasets -cnotcontains [string]$dataset) {
            $problems += "Unsupported dataset '$dataset'. Allowed values: $($AllowedDatasets -join ', ')."
        }
    }

    if (@($VARIANTS | Select-Object -Unique).Count -ne @($VARIANTS).Count) {
        $problems += "VARIANTS contains duplicate entries."
    }
    if (@($DATASETS | Select-Object -Unique).Count -ne @($DATASETS).Count) {
        $problems += "DATASETS contains duplicate entries."
    }

    # Only the variant scripts that will actually be run must exist here.
    foreach ($variant in @($VARIANTS)) {
        $variantPath = Join-Path $TransformerDir "$variant.py"
        if (-not (Test-Path -LiteralPath $variantPath -PathType Leaf)) {
            $problems += "Missing variant script: $variantPath"
        }
        elseif ((Get-Item -LiteralPath $variantPath).Length -eq 0) {
            $problems += "Variant script is empty: $variantPath"
        }
    }

    try {
        $pythonCommand = Get-Command python -CommandType Application -ErrorAction Stop |
            Select-Object -First 1
        $script:PythonExe = $pythonCommand.Path
        Write-Host "[OK] Python: $($script:PythonExe)" -ForegroundColor Green
    }
    catch {
        $problems += "Python was not found on PATH. Activate the Conda environment and try again."
    }

    if ($null -ne $script:PythonExe) {
        $cudaExitCode = -1
        $cudaOutput = @()
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $cudaOutput = @(
                & $script:PythonExe -c `
                    "import torch, sys; available = torch.cuda.is_available(); print(available); sys.exit(0 if available else 1)" `
                    2>&1
            )
            $cudaExitCode = $LASTEXITCODE
        }
        catch {
            $cudaOutput = @($_)
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        $cudaText = ($cudaOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
        if ($cudaExitCode -ne 0) {
            $problems += "CUDA is not visible to the active Python environment. Probe output: $cudaText"
        }
        else {
            Write-Host "[OK] torch.cuda.is_available(): True" -ForegroundColor Green
        }
    }

    $datasetLayouts = @{
        "xsum" = @{
            Directory = Join-Path $RepoRoot "xsum\data"
        }
        "cnn_dailymail" = @{
            Directory = Join-Path $RepoRoot "cnn_dailymail\3.0.0"
        }
    }

    foreach ($dataset in @($DATASETS)) {
        if ($AllowedDatasets -cnotcontains [string]$dataset) {
            continue
        }

        $datasetDir = $datasetLayouts[$dataset].Directory
        if (-not (Test-Path -LiteralPath $datasetDir -PathType Container)) {
            $problems += "Required local dataset directory is missing for '$dataset': $datasetDir"
            continue
        }

        $trainFiles = @(
            Get-ChildItem -LiteralPath $datasetDir -Filter "train-*.parquet" -File -ErrorAction SilentlyContinue
        )
        $testFiles = @(
            Get-ChildItem -LiteralPath $datasetDir -Filter "test-*.parquet" -File -ErrorAction SilentlyContinue
        )

        if ($trainFiles.Count -eq 0) {
            $problems += "No train-*.parquet files found for '$dataset' under $datasetDir"
        }
        if ($testFiles.Count -eq 0) {
            $problems += "No test-*.parquet files found for '$dataset' under $datasetDir"
        }
        if ($trainFiles.Count -gt 0 -and $testFiles.Count -gt 0) {
            Write-Host (
                "[OK] Local dataset: {0} -> {1} ({2} train shard(s), {3} test shard(s))" -f `
                    $dataset, $datasetDir, $trainFiles.Count, $testFiles.Count
            ) -ForegroundColor Green
        }
    }

    if ($problems.Count -gt 0) {
        Write-Host ""
        Write-Host "PREFLIGHT FAILED" -ForegroundColor Red
        foreach ($problem in $problems) {
            Write-Host "  - $problem" -ForegroundColor Red
        }
        return $false
    }

    Write-Host "[OK] All preflight checks passed." -ForegroundColor Green
    return $true
}

Write-Host "Transformer local seed sweep (cross-seed std)" -ForegroundColor Cyan
Write-Host "Repository:   $RepoRoot"
Write-Host "Seeds:        $($SEEDS -join ', ')"
Write-Host "Train sample: $TRAIN_SAMPLE"
Write-Host "Test sample:  $SAMPLE"
Write-Host "Variants:     $($VARIANTS -join ', ')"
Write-Host "Datasets:     $($DATASETS -join ', ')"
Write-Host "Device:       $DEVICE"
Write-Host "Force:        $Force"
Write-Host ""

if (-not (Invoke-Preflight)) {
    exit 2
}

if ($WhatIfPreference) {
    Write-Host ""
    Write-Host "WhatIf mode: preflight ran, but no directories, models, summaries, results, or logs will be written." -ForegroundColor Yellow
}
else {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

foreach ($seed in @($SEEDS)) {
    $seedInt = [int]$seed

    Write-Host ""
    Write-Host ("#" * 88) -ForegroundColor Magenta
    Write-Host "# SEED $seedInt" -ForegroundColor Magenta

    # Per-seed roots keep each seed's summaries and metrics isolated so the
    # seed-invariant output filenames never collide across seeds.
    $seedSummariesRoot = Join-Path $TransformerDir ("summaries_seed{0}" -f $seedInt)
    $seedResultsRoot = Join-Path $TransformerDir ("results_seed{0}" -f $seedInt)

    foreach ($variant in @($VARIANTS)) {
        $variantScript = Join-Path $TransformerDir "$variant.py"
        $summaryDir = Join-Path $seedSummariesRoot $variant
        $resultDir = Join-Path $seedResultsRoot $variant
        $evaluationLog = Join-Path $resultDir "evaluation.log"
        $evaluationCsv = Join-Path $resultDir "evaluation.csv"
        $variantRunAttempted = $false

        Write-Host ""
        Write-Host ("#" * 88) -ForegroundColor DarkCyan
        Write-Host "# Seed $seedInt / Variant $variant" -ForegroundColor DarkCyan

        foreach ($dataset in @($DATASETS)) {
            $summaryName = "Transformer_{0}_{1}_{2}_summaries.jsonl" -f $variant, $dataset, $SAMPLE
            $summaryPath = Join-Path $summaryDir $summaryName
            $runLog = Join-Path $LogDir ("{0}_{1}_{2}_seed{3}_all.log" -f $variant, $dataset, $SAMPLE, $seedInt)

            if ((-not $Force) -and (Test-NonEmptyFile -Path $summaryPath)) {
                Add-SkippedRun `
                    -Seed $seedInt `
                    -Variant $variant `
                    -Dataset $dataset `
                    -Task "all" `
                    -Reason "non-empty summary already exists: $summaryPath" `
                    -LogFile $runLog
                continue
            }

            $arguments = @(
                $variantScript,
                "--task", "all",
                "--dataset", [string]$dataset,
                "--train_sample", [string]$TRAIN_SAMPLE,
                "--sample", [string]$SAMPLE,
                "--seed", [string]$seedInt,
                "--device", [string]$DEVICE,
                "--output_dir", $summaryDir
            )

            # An empty leftover is not "done", but E*.py still requires
            # --overwrite merely because the path exists. Overwriting an empty
            # file is safe.
            if ($Force -or (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
                $arguments += "--overwrite"
            }

            $variantRunAttempted = $true
            $null = Invoke-LoggedRun `
                -Seed $seedInt `
                -Variant $variant `
                -Dataset $dataset `
                -Task "all" `
                -Arguments $arguments `
                -ExpectedFiles @($summaryPath) `
                -LogFile $runLog
        }

        $evaluationRunLog = Join-Path $LogDir ("{0}_seed{1}_all_evaluate.log" -f $variant, $seedInt)
        $evaluationOutputs = @($evaluationLog, $evaluationCsv)

        if (
            (-not $Force) -and
            (-not $variantRunAttempted) -and
            (Test-AllNonEmptyFiles -Paths $evaluationOutputs)
        ) {
            Add-SkippedRun `
                -Seed $seedInt `
                -Variant $variant `
                -Dataset "all" `
                -Task "evaluate" `
                -Reason "non-empty evaluation log and CSV already exist: $resultDir" `
                -LogFile $evaluationRunLog
            continue
        }

        $evaluationArguments = @(
            $variantScript,
            "--task", "evaluate",
            "--device", [string]$DEVICE,
            "--output_dir", $summaryDir,
            "--log_path", $evaluationLog
        )
        if ($Force) {
            # Accepted by every E*.py CLI. Evaluation itself always refreshes
            # its metric log/CSV, but passing it keeps -Force behavior uniform.
            $evaluationArguments += "--overwrite"
        }

        $null = Invoke-LoggedRun `
            -Seed $seedInt `
            -Variant $variant `
            -Dataset "all" `
            -Task "evaluate" `
            -Arguments $evaluationArguments `
            -ExpectedFiles $evaluationOutputs `
            -LogFile $evaluationRunLog `
            -FailurePattern 'WARNING:\s+\d+\s+file\(s\)\s+failed'
    }
}

Write-Host ""
Write-Host ("=" * 88) -ForegroundColor Cyan
Write-Host "FINAL RUN SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 88) -ForegroundColor Cyan

$script:RunResults |
    Select-Object Seed, Variant, Dataset, Task, Status, Duration, LogFile |
    Sort-Object Seed, Variant, Dataset, Task |
    Format-Table -AutoSize -Wrap |
    Out-Host

$failedCount = @($script:RunResults | Where-Object { $_.Status -eq "FAILED" }).Count
$okCount = @($script:RunResults | Where-Object { $_.Status -eq "OK" }).Count
$skippedCount = @($script:RunResults | Where-Object { $_.Status -eq "SKIPPED" }).Count

Write-Host "OK: $okCount  FAILED: $failedCount  SKIPPED: $skippedCount"

if ($script:AnyFailed -or $failedCount -gt 0) {
    Write-Host "One or more runs failed. Review the FAILED rows and their log files." -ForegroundColor Red
    exit 1
}

Write-Host "Pipeline completed without failures." -ForegroundColor Green
exit 0