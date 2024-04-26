import shutil
from pathlib import Path
from functions import entry, predict, calc_diagonal_cm, upload_response

AWS_ACCESS_KEY_ID = ...
AWS_SECRET_ACCESS_KEY = ...
AWS_DEFAULT_REGION = "ap-northeast-2"
S3_BUCKET_NAME = "dev-drone-square-bucket"
PROJECT_ID = "16"

input_data = entry(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    S3_BUCKET_NAME,
    PROJECT_ID,
)

labels = predict(target_dir=input_data["tiles"])


def severity_level(cm: float):
    if cm <= 30:
        return 1
    elif 30 < cm <= 60:
        return 2
    elif 60 < cm <= 100:
        return 3
    else:
        return 4


response = []
for label in labels:
    severities = []
    for boundary in label["boundaries"]:
        _, _, width, height, conf = boundary
        diagonal_cm = calc_diagonal_cm(width, height, input_data["gsd"])
        severities.append(severity_level(diagonal_cm))

    def calc_percentage(grade: int):
        percent = severities.count(grade) / len(severities) * 100
        return f"{round(percent)}%"

    result = {
        "point": list(label["coords"]),
        "grade": max(severities),
        "percentage": {
            "grade1": calc_percentage(1),
            "grade2": calc_percentage(2),
            "grade3": calc_percentage(3),
            "grade4": calc_percentage(4),
        },
    }

    response.append(result)

upload_response(PROJECT_ID, response, input_data["s3_client"])

work_dir = Path(__file__).parent / f"tiles/{PROJECT_ID}"
shutil.rmtree(work_dir, ignore_errors=True)
