param(
    [int]$Episodes = 50,
    [switch]$IncludePPO,
    [switch]$Include1M,
    [switch]$ProgressBar
)

$ErrorActionPreference = "Stop"

function Run-Step($Description, $Command) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Description
    Write-Host "============================================================"
    Write-Host $Command
    Invoke-Expression $Command
}

$progressArg = ""
if ($ProgressBar) { $progressArg = " --progress-bar" }

New-Item -ItemType Directory -Force -Path models, results, logs | Out-Null

Run-Step "Random baseline evaluation" "python src/random_baseline_ale.py --episodes $Episodes --output results/random_baseline_50ep.csv"

Run-Step "Train DQN 100k" "python src/train_dqn_ale_boxing.py --timesteps 100000 --model-out models/dqn_boxing_100k.zip --log-dir logs/dqn_boxing_100k$progressArg"
Run-Step "Evaluate DQN 100k" "python src/evaluate_agent.py --algo dqn --model models/dqn_boxing_100k.zip --episodes $Episodes --output results/dqn_evaluation_100k.csv --label 'DQN 100k'"

Run-Step "Train DQN 250k" "python src/train_dqn_ale_boxing.py --timesteps 250000 --model-out models/dqn_boxing_250k.zip --log-dir logs/dqn_boxing_250k$progressArg"
Run-Step "Evaluate DQN 250k" "python src/evaluate_agent.py --algo dqn --model models/dqn_boxing_250k.zip --episodes $Episodes --output results/dqn_evaluation_250k.csv --label 'DQN 250k'"

Run-Step "Train DQN 500k" "python src/train_dqn_ale_boxing.py --timesteps 500000 --model-out models/dqn_boxing_500k.zip --log-dir logs/dqn_boxing_500k$progressArg"
Run-Step "Evaluate DQN 500k" "python src/evaluate_agent.py --algo dqn --model models/dqn_boxing_500k.zip --episodes $Episodes --output results/dqn_evaluation_500k.csv --label 'DQN 500k'"

if ($Include1M) {
    Run-Step "Train DQN 1M" "python src/train_dqn_ale_boxing.py --timesteps 1000000 --model-out models/dqn_boxing_1000k.zip --log-dir logs/dqn_boxing_1000k$progressArg"
    Run-Step "Evaluate DQN 1M" "python src/evaluate_agent.py --algo dqn --model models/dqn_boxing_1000k.zip --episodes $Episodes --output results/dqn_evaluation_1000k.csv --label 'DQN 1000k'"
}

if ($IncludePPO) {
    Run-Step "Train PPO 500k" "python src/train_ppo_ale_boxing.py --timesteps 500000 --model-out models/ppo_boxing_500k.zip --log-dir logs/ppo_boxing_500k$progressArg"
    Run-Step "Evaluate PPO 500k" "python src/evaluate_agent.py --algo ppo --model models/ppo_boxing_500k.zip --episodes $Episodes --output results/ppo_evaluation_500k.csv --label 'PPO 500k'"
}

Run-Step "Aggregate results" "python src/aggregate_results.py"
Run-Step "Create plots" "python src/plot_results.py"

Write-Host ""
Write-Host "Done. Check results/professional_summary.csv and generated PNG plots in results/."
