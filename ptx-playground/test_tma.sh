#!/bin/bash
# Test script for TMA vs cp.async comparison

set -e

echo "========================================="
echo "PTX Tensor Core GEMM - TMA Test Suite"
echo "========================================="
echo ""

# Detect GPU
echo "Detecting GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU_NAME"
echo ""

# Determine architecture
if [[ $GPU_NAME == *"H100"* ]] || [[ $GPU_NAME == *"H200"* ]]; then
    ARCH="hopper"
    echo "Architecture: Hopper (sm_90) - TMA supported!"
elif [[ $GPU_NAME == *"A100"* ]] || [[ $GPU_NAME == *"RTX 40"* ]] || [[ $GPU_NAME == *"RTX 30"* ]]; then
    ARCH="ampere"
    echo "Architecture: Ampere (sm_80) - cp.async only"
else
    ARCH="unknown"
    echo "Architecture: Unknown - attempting Ampere build"
fi
echo ""

# Compile
echo "========================================="
echo "Compiling..."
echo "========================================="

if [ "$ARCH" == "hopper" ]; then
    echo "Building for Hopper (with TMA)..."
    make large_tma_hopper
    BINARY="./large_tma_hopper"
else
    echo "Building for Ampere (cp.async)..."
    make large_tma
    BINARY="./large_tma"
fi

echo ""
echo "========================================="
echo "Running Tests"
echo "========================================="
echo ""

# Test 1: cp.async
echo "--- Test 1: cp.async (Ampere compatible) ---"
$BINARY 0
echo ""

# Test 2: TMA (if available)
if [ "$ARCH" == "hopper" ]; then
    echo "--- Test 2: TMA (Hopper only) ---"
    $BINARY 1
    echo ""
else
    echo "--- Test 2: TMA (skipped - requires Hopper) ---"
    echo "TMA is not available on this GPU"
    echo ""
fi

echo "========================================="
echo "All tests completed!"
echo "========================================="

