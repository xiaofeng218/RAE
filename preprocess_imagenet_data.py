import os
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm
import glob

# 输入目录：你的 parquet 文件所在的路径
parquet_root = "/data/dataset/imagenet-1k/data/"

# 输出目录：转换后生成 ImageFolder 的地方
output_root = "/data/dataset/imagenet-1k/train/"

os.makedirs(output_root, exist_ok=True)

# 找到所有 parquet 文件
parquet_files = sorted(glob.glob(os.path.join(parquet_root, "*.parquet")))
print(f"Found {len(parquet_files)} parquet files.")

for parquet_path in parquet_files:
    print(f"\nProcessing {parquet_path} ...")

    # 加载 parquet 数据
    df = pd.read_parquet(parquet_path)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_bytes = row["image"]["bytes"]
        label = row["label"]

        # 解码
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 输出目录 (label 作为文件夹)
        label_dir = os.path.join(output_root, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # 确保每个文件不会冲突：使用全局编号
        filename = f"{os.path.basename(parquet_path)}_{idx}.jpg"

        # 保存
        img.save(os.path.join(label_dir, filename))

print("\nAll parquet files processed successfully!")
