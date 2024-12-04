pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/drive/folders/1XUAQa4prwJzOu7NBlpPzE1oW_wLvgUwj --folder

retrain_flag=""
if [ "$1" == "--retrain" ]; then
    retrain_flag="--retrain"
fi

python -m src.reproduc --folder "Best" --upload_template "TestSet_SubmissionTemplate/upload(no answer).csv" $retrain_flag