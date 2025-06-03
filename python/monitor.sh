#!/bin/bash

echo "===================================="
echo "TETRIS AI - TRAINING MONITOR"
echo "===================================="

# Function to display training status
show_status() {
    echo "Training Status - $(date)"
    echo "================================"
    
    # Show running Python processes
    echo "Running training processes:"
    ps aux | grep "train_tetris.py" | grep -v grep || echo "No training processes found"
    
    echo ""
    
    # Show latest models
    echo "Latest models:"
    if ls -t models/*.pth 2>/dev/null | head -5; then
        echo ""
    else
        echo "No models found"
    fi
    
    # Show TensorBoard logs
    echo "Recent TensorBoard logs:"
    if ls -t runs/ 2>/dev/null | head -5; then
        echo ""
    else
        echo "No logs found"
    fi
    
    # Show disk usage
    echo "Disk usage:"
    du -sh models/ runs/ logs/ 2>/dev/null || echo "No data directories found"
    
    echo "================================"
}

# Main monitoring loop
if [ "$1" = "--watch" ]; then
    echo "Monitoring mode - Press Ctrl+C to exit"
    while true; do
        clear
        show_status
        sleep 10
    done
else
    show_status
    echo ""
    echo "For continuous monitoring, run: ./monitor.sh --watch"
fi