import numpy as np

import torch
import fpsample

def load_ply(filename, intensity=True):
    """
    .ply ファイルを読み込み、
    点群(x, y, z, intensity)の NumPy 配列を返す。
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            header_index = None
            for i, line in enumerate(lines):
                if 'end_header' in line:
                    header_index = i
                    break
            if header_index is None:
                raise ValueError("PLYファイルのヘッダが正しく読み込めませんでした。")
            
            # ヘッダ以降の行を読み込み
            points = np.array([list(map(float, l.split())) for l in lines[header_index+1:]])
            #print('shape:', points.shape)
            xyz_points = points[:, :3]
            if not intensity:
                return xyz_points
            else:
                # [x, y, z, intensity] の形に整形
                # intensity が最後の列にあると仮定 (points[:, -1])
                points = np.concatenate([xyz_points, points[:, -1].reshape(-1, 1)], axis=1)
                return points
            
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        # 必要に応じて sys.exit(1) などで終了するか、Noneを返す
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました（load_ply）: {e}")
        return None

    
def write_ply(filename, pcd):
    """
    点群データを PLY ファイルとして保存する関数。
    出力フォーマットは以下の通り:
      property float32 x
      property float32 y
      property float32 z
      property uint8 r
      property uint8 g
      property uint8 b
      property float32 i
      
    数値の桁指定:
      - x, y, z: 小数点以下4桁まで
      - r, g, b: 整数 (常に 0)
      - i: 小数点以下2桁まで

    例:
      -1334.0197 -1060.7484 1785.6458 0 0 0 0.16

    入力:
      pcd: (N,3) または (N,4) の numpy 配列または torch.Tensor
           4列目が存在する場合は intensity として使用、なければ 0 とする。
    """
    if pcd is None:
        print(f"保存対象がNoneのためスキップ: {filename}")
        return 
    if isinstance(pcd, torch.Tensor):
        pcd_np = pcd.cpu().numpy()
    else:
        pcd_np = np.asarray(pcd)

    #入力形状のチェック(N,3)or(N,4)
    if pcd_np.ndim != 2 or pcd_np.shape[1] not in (3, 4):
        raise ValueError("pcdは(N, 3)また(N, 4)の形状である必要がある")
    
    #座標はfloat32として取得
    xyz = pcd_np[:, :3].astype(np.float32)

    #輝度値の取得
    if pcd_np.shape[1] == 4:
        intensity = pcd_np[:, 3].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((pcd_np.shape[0], 1), dtype = np.float32)
    
    #rgbはuint8の0として生成
    rgb = np.zeros((pcd_np.shape[0], 3), dtype = np.uint8)

    #x, y, z, r, g, b, iの順にデータを結合
    data = np.hstack((xyz, rgb, intensity))
    #plyヘッダの作成
    header = f"""ply
                format ascii 1.0
                element vertex {data.shape[0]}
                property float32 x
                property float32 y
                property float32 z
                property uint8 r
                property uint8 g
                property uint8 b
                property float32 i
                end_header
                """
    # ファイルにヘッダーと各点のデータを書き出す
    with open(filename, "w") as f:
        f.write(header)
        for row in data:
            # 書式: x,y,z は小数点以下4桁、i は小数点以下2桁で出力
            f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])} {int(row[4])} {int(row[5])} {row[6]:.2f}\n")


def fps_downsample(pcd, downsample_point, intensity = False) -> np.ndarray:
    fps_index = fpsample.fps_sampling(pcd, downsample_point)
    downsampled_pcd = pcd[fps_index][:, :3]
    if intensity:
        i = pcd[fps_index][:, 3].reshape(-1, 1)
        downsampled_pcd = np.concatenate([downsampled_pcd, i], axis=1)
    return downsampled_pcd

def normalize_pc(pointcloud: np.ndarray, intensity=False) -> np.ndarray:
    if intensity:
        i = pointcloud[:, 3].reshape(-1, 1)
        max_val = np.max(np.abs(pointcloud[:, :3]))
        norm_pointcloud = pointcloud / max_val
        norm_pointcloud = np.concatenate([norm_pointcloud, i], axis=1)
    else:
        # 点群全体の各座標の絶対値の最大値を計算
        max_val = np.max(np.abs(pointcloud))
        # 各点を max_val で割るだけのスケーリング（平行移動は行わない）
        norm_pointcloud = pointcloud / max_val
    return norm_pointcloud

def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_loss(T_pred, T_true):
    #回転行列得る
    R_pred = T_pred[:3, :3]
    R_true = T_true[:3, :3]

    R_pred_inv = np.linalg.inv(R_pred)
    loss_matrix = np.dot(R_pred_inv, R_true)
    R_loss = np.linalg.norm(loss_matrix - np.eye(3))

    #ｈ並進行列を得る
    t_pred = T_pred[:3, 3]
    t_true = T_true[:3, 3]
    print(f"t_pred: {t_pred}")
    print(f"t_ture: {t_true}")
    translation_loss = 1 - cos_similarity(t_pred, t_true)
    
    print(f"R_loss: {R_loss:.3f}")
    print(f"Translation loss: {translation_loss:.3f}")
    loss_value = R_loss + translation_loss
    return loss_value 

def compute_simple_loss(T_pred, T_true):
    T_pred_inv = np.linalg.inv(T_pred)
    loss_matrix = np.dot(T_pred_inv, T_true)
    loss_value = np.linalg.norm(loss_matrix - np.eye(4))
    return loss_value


def compute_transformation_errors(T_pred, T_true):
    R_pred = T_pred[:3, :3]
    R_true = T_true[:3, :3]
    R_pred_inv = np.linalg.inv(R_pred)
    loss_matrix = np.dot(R_pred_inv, R_true)
    R_loss = np.linalg.norm(loss_matrix - np.eye(3))

    t_pred = T_pred[:3, 3]
    t_true = T_true[:3, 3]
    translation_loss = 1 - cos_similarity(t_pred, t_true)

    return R_loss, translation_loss