import cdsapi
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime
import pandas as pd

# Assuming all_samples, asc_steps, and desc_steps are already defined in the environment
# For demonstration, let's recreate them from the prompt's context:
import os
from datetime import datetime, timedelta

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

def retrieve_and_convert_era5(
    date_time: datetime,
    base_dir: str,
    surface_variables: list[str],
    upper_variables: list[str],
    pressure_levels: list[str],
    area: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Downloads ERA5 single-level and pressure-level data for the given date_time.
    2. Converts them to numpy arrays and returns:
       - surface_data: shape (len(surface_variables), lat, lon)
       - upper_data:   shape (len(upper_variables), len(pressure_levels), lat, lon)

    Files are saved under base_dir/YYYY-MM-DD-HH-MM/ as 'surface.nc', 'upper.nc',
    'input_surface.npy', and 'input_upper.npy'.
    """
    client = cdsapi.Client()

    # Prepare output directory
    forecast_dir = os.path.join(base_dir, date_time.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(forecast_dir, exist_ok=True)

    # # Paths for NetCDF
    # surface_nc = os.path.join(forecast_dir, 'surface.nc')
    # upper_nc   = os.path.join(forecast_dir, 'upper.nc')

    # # Download surface data
    # client.retrieve('reanalysis-era5-single-levels', {
    #     'product_type': 'reanalysis',
    #     'format': 'netcdf',
    #     'variable': surface_variables,
    #     'date': date_time.strftime("%Y-%m-%d"),
    #     'time': date_time.strftime("%H:%M"),
    #     'area': area,
    # }, surface_nc)

    # # Download upper-air data
    # client.retrieve('reanalysis-era5-pressure-levels', {
    #     'product_type': 'reanalysis',
    #     'format': 'netcdf',
    #     'variable': upper_variables,
    #     'pressure_level': pressure_levels,
    #     'date': date_time.strftime("%Y-%m-%d"),
    #     'time': date_time.strftime("%H:%M"),
    #     'area': area,
    # }, upper_nc)

    # # Convert the surface data to npy
    # surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
    # with nc.Dataset(os.path.join(forecast_dir , 'surface.nc')) as nc_file:
    #     surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
    #     surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
    #     surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
    #     surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
    # np.save(os.path.join(forecast_dir, 'input_surface.npy'), surface_data)

    # #Convert the upper air data to npy
    # upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
    # with nc.Dataset(os.path.join(forecast_dir , 'upper.nc')) as nc_file:
    #     upper_data[0] = (nc_file.variables['z'][:]).astype(np.float32)
    #     upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
    #     upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
    #     upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
    #     upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
    # np.save(os.path.join(forecast_dir, 'input_upper.npy'), upper_data)


# —— 1) 把 filename 轉成 datetime —— 
def filename_to_datetime(fn: str) -> datetime:
    # fn 範例: "1980031500.npz" -> 取前 10 碼 "1980031500"
    return datetime.strptime(fn[:10], "%Y%m%d%H")

df['date_time'] = sorted(df['filename'].apply(filename_to_datetime))

surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

# Area to download
area = [90, 0, -90, 359.75]

# Pressure levels required
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

# 只取 2020 年的時間點
df2020 = df[df['date_time'].dt.year == 2024]

# —— 3) 迴圈呼叫下載函式 —— 
for dt in df2020['date_time']:
     # 這裡把 dt 轉成 "1980-03-15-00-00" 這種格式
    ts = dt.strftime("%Y-%m-%d-%H-%M")
    #forecast_dir = os.path.join(os.getcwd(), "forecasts", ts)
    forecast_dir = "forecast"
    try:
        #如果目標資料夾存在就跳過
        if os.path.exists(os.path.join(forecast_dir, ts)):
            print(f"Directory {os.path.join(forecast_dir, ts)} already exists. Skipping download.")
            continue
        else:
            os.makedirs(forecast_dir, exist_ok=True)
            retrieve_and_convert_era5(
            date_time=dt,
            base_dir=forecast_dir,
            surface_variables=surface_variables,
            upper_variables=upper_variables,
            pressure_levels=pressure_levels,
            area=area
        )
    except:
        continue
    # break