# === STEP 1: Login to Coder (opens browser) ===
Write-Host "=== Step 1: Login ===" -ForegroundColor Cyan
coder login https://coder.compute.isst.fraunhofer.de

# === STEP 2: Configure SSH ===
Write-Host "`n=== Step 2: Configure SSH ===" -ForegroundColor Cyan
coder config-ssh --yes

# === STEP 3: List workspaces to find yours ===
Write-Host "`n=== Step 3: Your workspaces ===" -ForegroundColor Cyan
coder list

# === STEP 4: Copy files to server ===
Write-Host "`n=== Step 4: Copying files to server ===" -ForegroundColor Cyan
Write-Host "Run this command (replace LLMtune with your workspace name from Step 3):" -ForegroundColor Yellow
Write-Host ""
Write-Host '  scp -r c:\Users\eggoni\Desktop\llm\*.py coder.LLMtune:/home/coder/llm/'
Write-Host '  scp -r c:\Users\eggoni\Desktop\llm\*.csv coder.LLMtune:/home/coder/llm/'
Write-Host '  scp -r c:\Users\eggoni\Desktop\llm\*.txt coder.LLMtune:/home/coder/llm/'
Write-Host '  scp -r c:\Users\eggoni\Desktop\llm\data coder.LLMtune:/home/coder/llm/'
Write-Host '  scp -r c:\Users\eggoni\Desktop\llm\classifiers coder.LLMtune:/home/coder/llm/'
Write-Host ""

# === STEP 5: SSH and run ===
Write-Host "=== Step 5: SSH into server and run ===" -ForegroundColor Cyan
Write-Host "  coder ssh LLMtune"
Write-Host ""
Write-Host "Then on the server run:" -ForegroundColor Green
Write-Host "  cd /home/coder/llm"
Write-Host "  pip install pandas requests"
Write-Host "  python diagnose_timeout.py"
Write-Host "  python classify_projects.py --model llama3.3:70b --limit 5 --verbose"
Write-Host ""
Write-Host "For full run:" -ForegroundColor Green
Write-Host "  nohup python classify_projects.py --model llama3.3:70b --verbose > run.log 2>&1 &"
Write-Host "  tail -f run.log"
Write-Host ""
Write-Host "To open VS Code remotely:" -ForegroundColor Green
Write-Host '  code --remote ssh-remote+coder.LLMtune /home/coder/llm'
