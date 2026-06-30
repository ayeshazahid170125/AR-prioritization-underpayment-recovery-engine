$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Running Project 02 AR recovery pipeline..." -ForegroundColor Cyan

$steps = @(
    "src\step_01_eda.py",
    "src\step01b_reload_headers.py",
    "src\step02_expected_payment.py",
    "src\step03_join_actual_expected.py",
    "src\step04a_null_detection.py",
    "src\step04b_outlier_detection.py",
    "src\step04c_cleaning.py",
    "src\step04d_benchmark_applicability.py",
    "src\step05_premodel_eda.py",
    "src\step06_feature_engineering.py",
    "src\step07_target_definition.py",
    "src\step08_collection_model.py",
    "src\step09_ar_priority_queue.py",
    "src\step10_isolation_forest_anomalies.py",
    "src\step11_underpayment_report.py",
    "src\step14_regression_validation.py"
)

foreach ($step in $steps) {
    Write-Host "`n=== Running $step ===" -ForegroundColor Yellow
    python $step
}

Write-Host "`nPipeline complete." -ForegroundColor Green
Write-Host "Next: run .\run_app.ps1 to start API + dashboard." -ForegroundColor Cyan
