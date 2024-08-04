# aws/aws_handler.py

import os
import boto3
import io

class AWSHandler:
    """
    Responsible for handling communication with Amazon AWS services.
    """

    def __init__(self, role_name: str, region_name: str = "eu-central-1") -> None:
        """
        Class constructor. Before calling it AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 
        and ACCOUNT_ID should be available in as environmental variables.

        Parameters:
            role_arn (str): Assumed role identification string.
            region_name (str): Region name to connect to.
        
        Raises:
            RuntimeError: If AWS credentials or account ID are not defined.
        """

        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        ACCOUNT_ID = os.getenv('ACCOUNT_ID')
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not ACCOUNT_ID:
            raise RuntimeError("AWS credentials or account ID not found in environment variables!")

        session = boto3.Session(aws_access_key_id = AWS_ACCESS_KEY_ID,
                                aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
        role_arn = f'arn:aws:iam::{ACCOUNT_ID}:role/{role_name}'

        assumed_role = session.client('sts').assume_role(RoleArn = role_arn, 
                                                         RoleSessionName = 'S3_bucket_user_session')
        credentials = assumed_role['Credentials']
        self.aws_s3_resource = boto3.client('s3', aws_access_key_id = credentials['AccessKeyId'],
                                            aws_secret_access_key = credentials['SecretAccessKey'],
                                            aws_session_token = credentials['SessionToken'],
                                            region_name = region_name)

    def upload_file_to_s3(self, bucket_name: str, file_path: str, desired_name: str = "") -> None:
        """
        Attempts to upload local file specified by path to S3 Amazon bucket.

        Parameters:
            bucket_name (str): String denoting bucket name.
            file_path (str): String representing file to the path that should be uploaded.
            desired_name (str): Desired name to be given to the file after being uploaded.
                If left unspecified, name does not change.
        
        Raises:
            RuntimeError: If approached problem during file uploading.
        """

        if desired_name == "":
            desired_name = file_path.split('/')[-1]
        try:
            self.aws_s3_resource.upload_file(file_path, bucket_name, desired_name)
        except Exception as e:
            raise RuntimeError(f"Did not managed to upload file! Original error: {e}")

    def upload_buffer_to_s3(self, bucket_name: str, buffer: io.StringIO, desired_name: str = "") -> None:
        """
        Attempts to upload buffer as file body directly to S3 Amazon bucket.

        Parameters:
            bucket_name (str): String denoting bucket name.
            buffer (io.StringIO): Buffer containing data that should be directly
                written to bucket.
            desired_name (str): Desired name to be given to the file after being uploaded.

        Raises:
            RuntimeError: If approached problem during file uploading.
        """

        try:
            self.aws_s3_resource.put_object(Bucket = bucket_name, Key = desired_name, 
                                            Body = buffer.getvalue())
        except Exception as e:
            raise RuntimeError(f"Did not managed to upload file! Original error: {e}")
