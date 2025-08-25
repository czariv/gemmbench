#!/bin/bash

passed=0
failed=0
total=0

for m in $(seq 128 1 144); do
	for k in $(seq 128 1 132); do
		for n in $(seq 1 1 48); do
			((total++))
			./gemmvbench $m $n $k 1 0
			if [ $? -eq 0 ]; then
				((passed++))
				echo "$m $n $k -- PASS"
			else
				((failed++))
				echo "$m $n $k -- FAILED"
			fi
		done
	done
done

echo "#$total Tests"
echo "Passed: $passed"
echo "Failed: $failed"
