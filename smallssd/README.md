To run this (assuming you have the `smallSSD` package installed):
1. Install the additional requirements in the [`../requirements.txt`](requirements.txt) file
2. Run [`end_to_end.py`](end_to_end.py)

This end to end script will first train a model in a fully-supervised fashion ("burn-in"), and then train it in a semi-supervised manner using a teacher-student method. Trained models will be stored (by default) in `lightning_logs`.

A [test script](test.py) is additionally provided to test models.

```bash
python end_to_end.py --model <FULLY_TRAINED_MODEL_TYPE>
```

We use [MLFLow](https://mlflow.org/) to track and deploy models; if you want to integrate this pipeline into MLFlow as well, the following environment variables will need to be set:

- MLFLOW_TRACKING_USERNAME
- MLFLOW_TRACKING_PASSWORD
- MLFLOW_TRACKING_URI

If you are saving your models to [S3](https://aws.amazon.com/s3/), you will additionally need:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
