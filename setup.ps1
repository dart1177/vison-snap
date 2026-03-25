# setup.ps1 - PowerShell setup script

Write-Host "Setting up Room Memory Project..." -ForegroundColor Green

# Create directories
$folders = @("maps", "captures", "test_images")
foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Name $folder | Out-Null
        Write-Host "Created: $folder" -ForegroundColor Cyan
    } else {
        Write-Host "Exists: $folder" -ForegroundColor Yellow
    }
}

# Check Python
Write-Host "`nChecking Python..." -ForegroundColor Green
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}

if ($python) {
    Write-Host "Python found: $($python.Source)" -ForegroundColor Cyan
    
    # Install requirements
    Write-Host "`nInstalling dependencies..." -ForegroundColor Green
    & $python.Source -m pip install --upgrade pip
    & $python.Source -m pip install opencv-python numpy pillow
    
    # Test import
    Write-Host "`nTesting imports..." -ForegroundColor Green
    $testCode = @"
import sys
print("Python version:", sys.version)
try:
    import cv2
    print("✅ OpenCV version:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV error:", e)
try:
    import numpy as np
    print("✅ NumPy version:", np.__version__)
except ImportError as e:
    print("❌ NumPy error:", e)
"@
    
    $testCode | & $python.Source -
    
} else {
    Write-Host "❌ Python not found! Please install Python 3.7+ from python.org" -ForegroundColor Red
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Run: python main.py" -ForegroundColor Yellow