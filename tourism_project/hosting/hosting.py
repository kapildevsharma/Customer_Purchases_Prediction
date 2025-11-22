from huggingface_hub import HfApi
import os

# https://huggingface.co/spaces/kapilmika/customer_purchases_prediction
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="kapilmika/customer-purchases-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
