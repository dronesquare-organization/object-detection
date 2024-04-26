import os, io, json, math, subprocess, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

from tms_encoder import TMSPathEncoder, Polygon, Point

s3 = boto3.client("s3")


class S3:
    def __init__(self, access_key, secret_key, region, bucket_name):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self.bucket_name = bucket_name

    def download(self, path) -> bytes:
        """오브젝트를 다운로드하여 bytes로 반환"""
        self.client.download_fileobj(
            self.bucket_name, str(path), buffer := io.BytesIO()
        )
        return buffer.getvalue()

    def upload(
        self,
        file: bytes,
        path: str,
        content_type: str = "binary/octet-stream",
        metadata: dict = {},
    ):
        self.client.upload_fileobj(
            io.BytesIO(file),
            self.bucket_name,
            str(path),
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": metadata,
            },
        )

    def download_file(self, target_path, save_path):
        """오브젝트를 다운로드하여 save_path로 저장"""
        with open(save_path, "wb") as file_obj:
            self.client.download_fileobj(self.bucket_name, str(target_path), file_obj)
        print(f"[S3.download_file] {target_path} -> {save_path}")

    def download_files(self, target_path_list, save_path_list):
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 10) as executor:
            futures = []
            for target_path, save_path in zip(target_path_list, save_path_list):
                future = executor.submit(
                    self.download_file,
                    target_path,
                    save_path,
                )
                futures.append(future)
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(
                        f"[S3.download_files] error occurred: [{exc.__class__}] {exc}"
                    )


def tms_gsd(latitude: float, zoom_level: int) -> float:
    """cm 단위의 픽셀 해상도 반환"""
    C = 40_075_016.686  # 적도 둘래
    gsd_m = (C * math.cos(math.radians(latitude))) / (2 ** (zoom_level + 8))
    return gsd_m * 100


def get_request(s3: S3, project_id: str | int) -> dict:
    """S3 에서 요청 json 다운로드 후 객체로 반환"""
    jsonb = s3.download(f"public/{project_id}/auto-detection/pothole/request.json")
    return json.loads(jsonb)


def get_tiles(
    s3: S3, tms_encoder: TMSPathEncoder, area_list: list, save_dir: Path | str
) -> list:
    """
    - 처리 대상 타일 이미지를 다운로드합니다.
    - tms_encoder: TMSPathEncoder 인스턴스
    - area_list: 처리 대상 구역들
    - return: 저장 경로 리스트
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tile_path_list = []
    for area in area_list:
        area_polygon = [tms_encoder.crs.lnglat_to_xy(lng, lat) for lng, lat in area]
        inner_tiles = tms_encoder.polygon2tms(
            Polygon(*[Point(x, y) for x, y in area_polygon])
        )
        tile_path_list.extend(inner_tiles)

    save_path_list = []
    for tile_path in tile_path_list:
        lng, lat = tms_encoder.tms2lnglat(tile_path)
        save_path_list.append(str(save_dir / f"{lng}_{lat}.png"))
    s3.download_files(tile_path_list, save_path_list)
    return save_path_list


def entry(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    S3_BUCKET_NAME,
    PROJECT_ID,
):
    s3 = S3(
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_DEFAULT_REGION,
        S3_BUCKET_NAME,
    )
    # ============================================
    request_json = get_request(s3, PROJECT_ID)
    gsd_cm = tms_gsd(
        latitude=sum(request_json["latitude"]) / 2,
        zoom_level=request_json["level"],
    )
    save_dir = f"tiles/{PROJECT_ID}"
    get_tiles(
        s3,
        tms_encoder=TMSPathEncoder(
            root=f"public/{PROJECT_ID}/manifold/orthomosaic_tiles",
            epsg=request_json["epsg"],
            level=request_json["level"],
        ),
        area_list=request_json["area"],
        save_dir=save_dir,
    )
    return {"gsd": gsd_cm, "tiles": save_dir, "s3_client": s3}


def predict(
    target_dir: str | Path,
    img_size: int = 256,
    weights: str | Path = "weights/pothole.pt",
):
    """
    - weight: 가중치 파일 경로
    - img_size: 이미지 크기
    - target: 대상 이미지들이 담긴 디렉토리 경로
    - return: 경위도 좌표와 해당 좌표에 위치한 타일에 있는 포트홀 바운더리들
        - 바운더리의 마지막 값은 신뢰도(0~1)임
    """
    root = Path(__file__).parent
    target_dir = Path(target_dir)
    dir_name = "predict"
    dir_path = target_dir / dir_name
    if dir_path.exists() and any(dir_path.iterdir()):
        # 디렉토리 내부의 모든 파일과 서브디렉토리 삭제
        for item in dir_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)  # 서브디렉토리 삭제
            else:
                item.unlink()  # 파일 삭제

    execute = ["python", str(root / "yolov9/detect.py")]
    options = (
        ["--weights", str(weights)]
        + ["--img", str(img_size)]
        + ["--source", str(target_dir / "*.png")]
        + ["--name", dir_name]
        + ["--project", str(target_dir)]
        + ["--conf-thres", "0.5"]  # 신뢰도 역치를 0.5로 상향조정
    )
    settings = ["--exist-ok", "--save-txt", "--save-conf"]
    subprocess.run(execute + options + settings)
    return [
        {
            "coords": tuple(map(float, file.stem.split("_"))),
            "boundaries": [
                line.split(" ")[1:] for line in file.read_text().split("\n") if line
            ],
        }
        for file in (target_dir / dir_name / "labels").iterdir()
        if file.is_file()
    ]


def calc_diagonal_cm(width, height, gsd):
    """YOLO 형식 라벨의 너비, 높이 값으로 대각선 길이를 계산합니다."""
    width_cm = float(width) * 256 * float(gsd)
    height_cm = float(height) * 256 * float(gsd)
    return math.sqrt(width_cm**2 + height_cm**2)


def upload_response(project_id, response: list, s3: S3):
    upload_path = f"public/{project_id}/auto-detection/pothole/result.json"
    json.dump(response, buffer := io.StringIO())
    s3.upload(buffer.getvalue().encode(), upload_path, content_type="application/json")
