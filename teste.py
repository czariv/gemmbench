import os
import difflib

before_dir = "before"
after_dir = "after"

# Ensure both directories exist
if not os.path.isdir(before_dir) or not os.path.isdir(after_dir):
    print("Both 'before' and 'after' folders must exist.")
    exit(1)

# Collect all CSV filenames in before/
files = sorted(f for f in os.listdir(before_dir) if f.endswith(".csv"))

for fname in files:
    before_path = os.path.join(before_dir, fname)
    after_path = os.path.join(after_dir, fname)

    if not os.path.exists(after_path):
        print(f"❌ Missing file in after/: {fname}")
        continue

    with open(before_path) as f1, open(after_path) as f2:
        before_lines = f1.readlines()
        after_lines = f2.readlines()

    if before_lines != after_lines:
        break
        print(f"🔍 Differences in {fname}:")
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"before/{fname}",
            tofile=f"after/{fname}",
            lineterm=""
        )
        for line in diff:
            print(line)
    else:
        print(f"✅ {fname} matches")
