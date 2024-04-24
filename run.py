from functions import entry, predict

input_data = entry(
    AWS_ACCESS_KEY_ID="...",
    AWS_SECRET_ACCESS_KEY="...",
    AWS_DEFAULT_REGION="ap-northeast-2",
    S3_BUCKET_NAME="dev-drone-square-bucket",
    PROJECT_ID="16",
)

labels = predict(target_dir=input_data["tiles"])
