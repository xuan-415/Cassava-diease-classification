import os, shutil, pandas as pd

def main():
    # 讀取csv文件，這裡假設文件第一行為標題
    df = pd.read_csv('train.csv')

    # 指定要保存圖片的目標資料夾
    target_folder = "dataset"
    path = "train_images"

    # 創建目標資料夾，如果不存在
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # 將圖像根據標籤分類
    for label in df['label'].unique():
        # 創建標籤對應的資料夾，如果不存在
        label_folder = os.path.join(target_folder, str(label))
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)
            
        # 選擇所有標籤為label的行
        mask = df['label'] == label
        rows = df.loc[mask]
        
        # 遍歷這些行，將圖像移動到對應的資料夾中
        for _, row in rows.iterrows():
            image_name = row['image_id']
            src_path = os.path.join(path, image_name)
            dst_path = os.path.join(label_folder, image_name)
            shutil.move(src_path, dst_path)

if __name__ == '__main__':
    main()
