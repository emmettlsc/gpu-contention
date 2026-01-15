# ALL LLM GEN - to run things like ncu and aggregate results


# run ncu kernel characterization and aggregate results to csv
#
# usage:
#   ./run_ncu_characterization.sh <image_path> [output_dir]
#
# example:
#   ./run_ncu_characterization.sh test_image.jpg
#   ./run_ncu_characterization.sh test_image.jpg results/

set -e

IMAGE_PATH="${1:?usage: $0 <image_path> [output_dir]}"
OUTPUT_DIR="${2:-ncu_results}"
BINARY="../../build/characterize_kernels"
NCU=/usr/local/cuda/bin/ncu

# find binary
if [ ! -f "$BINARY" ]; then
    BINARY="../build/characterize_kernels"
fi
if [ ! -f "$BINARY" ]; then
    echo "error: characterize_kernels binary not found"
    echo "build it first: cd build && cmake .. && make characterize_kernels"
    exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "error: image not found: $IMAGE_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCU_RAW="$OUTPUT_DIR/ncu_raw_${TIMESTAMP}.csv"
NCU_SUMMARY="$OUTPUT_DIR/ncu_summary_${TIMESTAMP}.csv"
NCU_LOG="$OUTPUT_DIR/ncu_log_${TIMESTAMP}.txt"

echo "=== ncu kernel characterization ==="
echo "image: $IMAGE_PATH"
echo "output: $OUTPUT_DIR"
echo ""

# check ncu is available
if ! command -v "$NCU" &> /dev/null; then
    echo "error: ncu (nsight compute) not found in PATH"
    echo "install with: apt install nsight-compute  (or add to PATH)"
    exit 1
fi

echo "running ncu (this should take 2-5 minutes)..."
echo ""

# run ncu with csv output
# --target-processes all ensures child processes are profiled
# note: -s/-c are GLOBAL launch counters (not per-kernel), so we don't use them.
# --ncu mode in characterize_kernels already limits to 3 runs per kernel,
# keeping total profiled launches small (~40 total across all kernels).
#
# metrics collected:
#   sm__throughput                - SM compute throughput (% of peak)
#   dram__throughput              - DRAM memory throughput (% of peak)
#   launch__registers_per_thread  - registers used per thread (resource footprint)
#   launch__shared_mem_per_block  - shared memory per block in bytes (resource footprint)
#   sm__warps_active              - achieved occupancy (% of peak active warps)
#   l1tex__t_bytes                - L1 cache throughput (% of peak)
#   lts__t_bytes                  - L2 cache throughput (% of peak)
"$NCU" --csv \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__shared_mem_per_block_nominal,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes.avg.pct_of_peak_sustained_active,lts__t_bytes.avg.pct_of_peak_sustained_active \
    --target-processes all \
    "$BINARY" --ncu "$IMAGE_PATH" \
    > "$NCU_RAW" 2>"$NCU_LOG"

echo "ncu complete. raw csv: $NCU_RAW"
echo ""

# aggregate per-kernel metrics to summary csv
echo "aggregating results..."

python3 - "$NCU_RAW" "$NCU_SUMMARY" << 'PYEOF'
import csv
import sys
import io
from collections import defaultdict

raw_path = sys.argv[1]
summary_path = sys.argv[2]

# per-kernel accumulator for all 7 metrics + grid/block info
def make_entry():
    return {
        "sm_pct": [], "dram_pct": [],
        "regs_per_thread": [], "smem_per_block": [],
        "occupancy_pct": [],
        "l1_pct": [], "l2_pct": [],
        "grid": "", "block": "",
    }

kernel_data = defaultdict(make_entry)

with open(raw_path, 'r') as f:
    lines = f.readlines()

# find the CSV header line (starts with "ID" or '"ID"')
header_idx = None
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith('"ID"') or stripped.startswith('ID,'):
        header_idx = i
        break

if header_idx is None:
    print("error: no ncu CSV header found (looking for line starting with 'ID')")
    print(f"file has {len(lines)} lines. first 5:")
    for line in lines[:5]:
        print(f"  {line.rstrip()}")
    sys.exit(1)

# parse from header onwards
csv_text = ''.join(lines[header_idx:])
reader = csv.DictReader(io.StringIO(csv_text))

print(f"detected csv columns: {reader.fieldnames}")

# metric name substring -> accumulator key
METRIC_MAP = {
    "sm__throughput":                    "sm_pct",
    "dram__throughput":                  "dram_pct",
    "launch__registers_per_thread":      "regs_per_thread",
    "launch__shared_mem_per_block":      "smem_per_block",
    "sm__warps_active":                  "occupancy_pct",
    "l1tex__t_bytes":                    "l1_pct",
    "lts__t_bytes":                      "l2_pct",
}

row_count = 0
for row in reader:
    kernel_name = row.get("Kernel Name", "").strip().strip('"')
    metric_name = row.get("Metric Name", "").strip().strip('"')
    metric_value = row.get("Metric Value", "0").strip().strip('"')

    if not kernel_name or not metric_name:
        continue

    try:
        val = float(metric_value.replace(",", ""))
    except ValueError:
        continue

    row_count += 1

    for substr, key in METRIC_MAP.items():
        if substr in metric_name:
            kernel_data[kernel_name][key].append(val)
            break

    # capture grid/block info
    grid = row.get("Grid Size", "")
    block = row.get("Block Size", "")
    if grid:
        kernel_data[kernel_name]["grid"] = grid.strip('"')
    if block:
        kernel_data[kernel_name]["block"] = block.strip('"')

print(f"parsed {row_count} metric rows across {len(kernel_data)} unique kernels")

def avg(vals):
    return sum(vals) / len(vals) if vals else 0

def first_or(vals, default=0):
    """For metrics like registers that are constant per kernel, take the first value."""
    return vals[0] if vals else default

# write summary csv
with open(summary_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "kernel",
        "avg_sm_throughput_pct", "avg_dram_throughput_pct",
        "registers_per_thread", "shared_mem_per_block_bytes",
        "avg_occupancy_pct",
        "avg_l1_throughput_pct", "avg_l2_throughput_pct",
        "grid", "block", "bottleneck",
    ])

    for kernel_name in sorted(kernel_data.keys()):
        d = kernel_data[kernel_name]
        avg_sm = avg(d["sm_pct"])
        avg_dram = avg(d["dram_pct"])

        if not d["sm_pct"] and not d["dram_pct"]:
            continue

        if avg_dram > avg_sm * 1.5:
            bottleneck = "memory-bound"
        elif avg_sm > avg_dram * 1.5:
            bottleneck = "compute-bound"
        else:
            bottleneck = "balanced"

        writer.writerow([
            kernel_name,
            f"{avg_sm:.1f}", f"{avg_dram:.1f}",
            f"{first_or(d['regs_per_thread']):.0f}",
            f"{first_or(d['smem_per_block']):.0f}",
            f"{avg(d['occupancy_pct']):.1f}",
            f"{avg(d['l1_pct']):.1f}", f"{avg(d['l2_pct']):.1f}",
            d["grid"], d["block"], bottleneck,
        ])

# print summary table
print("")
hdr = (f"{'kernel':<45} {'sm%':>6} {'dram%':>6} {'regs':>5} {'smem':>7} "
       f"{'occ%':>6} {'L1%':>6} {'L2%':>6}  {'bottleneck':<15}")
print(hdr)
print("-" * len(hdr))

for kernel_name in sorted(kernel_data.keys()):
    d = kernel_data[kernel_name]
    if not d["sm_pct"] and not d["dram_pct"]:
        continue
    avg_sm = avg(d["sm_pct"])
    avg_dram = avg(d["dram_pct"])
    regs = first_or(d["regs_per_thread"])
    smem = first_or(d["smem_per_block"])
    occ = avg(d["occupancy_pct"])
    l1 = avg(d["l1_pct"])
    l2 = avg(d["l2_pct"])

    if avg_dram > avg_sm * 1.5:
        bottleneck = "memory-bound"
    elif avg_sm > avg_dram * 1.5:
        bottleneck = "compute-bound"
    else:
        bottleneck = "balanced"

    print(f"{kernel_name:<45} {avg_sm:>5.1f}% {avg_dram:>5.1f}% {regs:>5.0f} {smem:>6.0f}B "
          f"{occ:>5.1f}% {l1:>5.1f}% {l2:>5.1f}%  {bottleneck:<15}")

print("")
print(f"summary csv written to: {summary_path}")
PYEOF

echo ""
echo "=== done ==="
echo "files:"
echo "  raw ncu csv:    $NCU_RAW"
echo "  summary csv:    $NCU_SUMMARY"
echo "  ncu stderr log: $NCU_LOG"
