import math
import gradio as gr

# ----------------------------
# Ns 計算関数
# ----------------------------
def estimate_ns(total_mem_bytes, usage_ratio, nb):
    usable_mem = total_mem_bytes * usage_ratio
    raw_N = math.sqrt(usable_mem / 8)  # 1要素 = 8バイト（double）
    adjusted_N = int(raw_N // nb * nb)  # NB の倍数に切り下げ
    return adjusted_N

# ----------------------------
# P×Q 最適近似（平方根に近い因数分解）
# ----------------------------
def find_best_pq(nprocs):
    best_pair = (1, nprocs)
    min_diff = float('inf')
    for i in range(1, int(math.sqrt(nprocs)) + 1):
        if nprocs % i == 0:
            j = nprocs // i
            diff = abs(i - j)
            if diff < min_diff:
                min_diff = diff
                best_pair = (i, j)
    return best_pair

# ----------------------------
# メイン処理（Gradio 用ラッパー）
# ----------------------------
def optimize_params(num_gpus, mem_per_gpu_mem, mem_utilization, nb):
    # バイトに変換
    total_mem_bytes = num_gpus * mem_per_gpu_mem * (1024**2)
    # Ns 計算
    ns = estimate_ns(total_mem_bytes, mem_utilization, nb)
    # P×Q 計算
    p, q = find_best_pq(int(num_gpus))
    # 結果を文字列で返す
    return (
        f"Ns = {ns}\n"
        f"NBs = {nb}\n"
        f"Ps = {p}\n"
        f"Qs = {q}\n\n"
        "※NB, Ns, P, Q に応じて HPL.dat を生成してください。"
    )

# ----------------------------
# Gradio インターフェイス定義
# ----------------------------
iface = gr.Interface(
    fn=optimize_params,
    inputs=[
        gr.Number(label="GPU数", value=504, precision=0),
        gr.Number(label="1枚あたりのGPUメモリ容量（MiB）", value=183359),
        gr.Slider(label="メモリ使用率", minimum=0.0, maximum=1.0, step=0.01, value=0.95),
        gr.Number(label="ブロックサイズ（NBs）", value=2048, precision=0),
    ],
    outputs=gr.Textbox(label="最適化結果"),
    title="HPL Ns, Ps, Qs 計算ツール",
    description="GPUクラスタのメモリ情報から HPL の Ns, Ps, Qs を自動計算します。",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
