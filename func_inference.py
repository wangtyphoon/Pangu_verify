import os
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime, timedelta
import pandas as pd
def run_pangu_inference(
    start_time: datetime,
    end_time: datetime,
    model_paths: dict,
    base_dir: str = None,
    use_gpu: bool = True,
    intra_threads: int = 30
) -> None:
    """
    Run Pangu-Weather ONNX inference from start_time to end_time.

    每次往前推最大的 interval，直到 current_time >= end_time。
    """
    # 1) 設定目錄
    if base_dir is None:
        base_dir = os.getcwd()
    forecasts_dir = os.path.join(base_dir, 'forecasts', start_time.strftime("%Y-%m-%d-%H-%M"))
    results_dir = os.path.join(
        base_dir, 'results',
        f"{start_time.strftime('%Y-%m-%d-%H-%M')}to{end_time.strftime('%Y-%m-%d-%H-%M')}"
    )
    if os.path.exists(results_dir):
        return
    os.makedirs(results_dir, exist_ok=True)

    # 2) 初始化
    current_time = start_time
    last_model = None
    ort_session = None
    is_first = True

    # 3) 主迴圈：每次向前推階段
    while current_time < end_time:
        remaining_hours = (end_time - current_time).total_seconds() / 3600

        # 選擇最大的 interval
        for interval in sorted(model_paths.keys(), reverse=True):
            if remaining_hours >= interval:
                break
        model_file = model_paths[interval]
        print(f"[{current_time}] → +{interval}h  使用模型：{os.path.basename(model_file)}")

        # 4) 若模型換了，重建 ONNX session
        if model_file != last_model:
            sess_opts = ort.SessionOptions()
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.enable_mem_pattern = False
            sess_opts.enable_mem_reuse = False
            sess_opts.intra_op_num_threads = intra_threads

            providers = ['CPUExecutionProvider']
            if use_gpu:
                cuda_opts = {'arena_extend_strategy':'kSameAsRequested'}
                providers = [('CUDAExecutionProvider', cuda_opts)]

            ort_session = ort.InferenceSession(model_file, sess_options=sess_opts, providers=providers)
            last_model = model_file

        # 5) 載入輸入
        if is_first:
            upper = np.load(os.path.join(forecasts_dir, 'input_upper.npy')).astype(np.float32)
            surface = np.load(os.path.join(forecasts_dir, 'input_surface.npy')).astype(np.float32)
            is_first = False
        else:
            prev_stamp = current_time.strftime("%Y-%m-%d-%H-%M")
            upper = np.load(os.path.join(results_dir, f'output_upper_{prev_stamp}.npy')).astype(np.float32)
            surface = np.load(os.path.join(results_dir, f'output_surface_{prev_stamp}.npy')).astype(np.float32)

        # 6) 推斷 + 存檔
        outputs = ort_session.run(None, {'input': upper, 'input_surface': surface})
        out_upper, out_surface = outputs

        # 7) 更新時間，再存成新的檔名
        current_time += timedelta(hours=interval)
        stamp = current_time.strftime("%Y-%m-%d-%H-%M")
        np.save(os.path.join(results_dir, f'output_upper_{stamp}.npy'), out_upper)
        np.save(os.path.join(results_dir, f'output_surface_{stamp}.npy'), out_surface)

    print("Inference complete.")

reference_path = r"D:\RDL_TC_MODEL\npz\test\prs"
all_samples = sorted([f for f in os.listdir(reference_path) if f.endswith('.npz')])

def consecutive_6h_steps(files, ascending=True):
    files_sorted = sorted(files, reverse=not ascending)
    times = [datetime.strptime(f[:10], "%Y%m%d%H") for f in files_sorted]
    step_list = [0]
    for i in range(1, len(times)):
        delta = times[i] - times[i - 1]
        target = timedelta(hours=6) if ascending else timedelta(hours=-6)
        if delta == target:
            step_list.append(step_list[-1] + 1)
        else:
            step_list.append(0)
    return step_list if ascending else step_list[::-1]

asc_steps = consecutive_6h_steps(all_samples, ascending=True)
desc_steps = consecutive_6h_steps(all_samples, ascending=False)

# Create the DataFrame
df = pd.DataFrame({
    'filename': all_samples,
    'asc_steps': asc_steps,
    'desc_steps': desc_steps
})
# —— 1) 把 filename 轉成 datetime —— 
def filename_to_datetime(fn: str) -> datetime:
    # fn 範例: "1980031500.npz" -> 取前 10 碼 "1980031500"
    return datetime.strptime(fn[:10], "%Y%m%d%H")
df['date_time'] = df['filename'].apply(filename_to_datetime)
fourcast_steps = 1  # Define the number of steps for fourcast
mask2024 = (
    (df['date_time'].dt.year == 2024) &
    (df['asc_steps']  >= fourcast_steps) &
    (df['desc_steps'] >= fourcast_steps)
)

df2024 = df.loc[mask2024].copy()
model_paths = {
        24: 'models/pangu_weather_24.onnx',
        6:  'models/pangu_weather_6.onnx',
        3:  'models/pangu_weather_3.onnx',
        1:  'models/pangu_weather_1.onnx'
    }
# 假設 df2020['date_time'] 已經準備好，一次往後推 6 小時
for dt in df2024['date_time']:
    run_pangu_inference(
        start_time=dt,
        end_time=dt + timedelta(hours=6),
        model_paths=model_paths,
        base_dir=r"C:\Users\王子祐\Documents\Pangu-Weather-ReadyToGo",  # 或其它根目錄
        use_gpu=True
    )
    # break