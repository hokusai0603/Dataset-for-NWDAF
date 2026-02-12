
$ErrorActionPreference = "Stop"

Write-Host "Starting repository update..." -ForegroundColor Cyan

# Add all changes (respecting .gitignore)
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto-update: $timestamp"
    
    Write-Host "Pushing changes to remote..." -ForegroundColor Cyan
    git push origin main
    
    Write-Host "Update complete!" -ForegroundColor Green
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

Pause
