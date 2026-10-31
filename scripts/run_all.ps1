param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference = 'Stop'
foreach ($task in @('binary', 'counting')) {
  if ($task -eq 'binary') { $seed = 40 } else { $seed = 42 }
  foreach ($method in @('naive', 'random_dr', 'cdr')) {
    python -m cvis_cdr.experiment train --manifest $Manifest --task $task --method $method --seed $seed --epochs 30 --input-size 224
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $task $method" }
  }
  python -m cvis_cdr.compare --manifest $Manifest --task $task --seed $seed
  if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $task" }
}
