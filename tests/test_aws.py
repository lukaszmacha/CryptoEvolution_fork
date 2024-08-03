# tests/test_aws.py

import os
from unittest.mock import patch
from moto import mock_aws
import boto3
from source.aws import AWSHandler
import tempfile

@mock_aws
@patch.dict(os.environ, {
    'AWS_ACCESS_KEY_ID': 'access_key',
    'AWS_SECRET_ACCESS_KEY': 'secret_key',
    'ACCOUNT_ID': '123456789012'
})
def test_upload_file_to_s3():
    """
    Tests the upload_file_to_s3 method of AWSHandler.

    Verifies that method successfully uploads a file to an S3 bucket.
    It mocks AWS S3 to create a bucket and handle the file upload. Also,
    the open function is mocked to simulate file creation and writing.

    Asserts:
        The content of the uploaded file in the S3 bucket matches the expected content.
    """

    mocked_bucket_name = 'mocked-bucket'
    mocked_region = 'us-east-1'
    mocked_s3_client = boto3.client('s3', region_name = mocked_region)
    mocked_s3_client.create_bucket(Bucket = mocked_bucket_name)
        
    with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
        tmp_file.write(b'Test file content')
        mocked_file_path = tmp_file.name

    desired_name = 'file.txt'
    handler = AWSHandler('mocked_bucket-user-role')
    handler.upload_file_to_s3(mocked_bucket_name, mocked_file_path, desired_name)
    os.remove(mocked_file_path)

    mocked_s3_resource = boto3.resource('s3', region_name = mocked_region)
    object = mocked_s3_resource.Object(mocked_bucket_name, desired_name)
    file_content = object.get()['Body'].read().decode('utf-8')

    assert file_content == 'Test file content'