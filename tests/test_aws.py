# tests/test_aws.py

import os
import io
import tempfile
import boto3
from unittest.mock import patch
from moto import mock_aws

from source.aws import AWSHandler

TEST_CONTENT = "This is a test file content."
TEST_FILE_NAME = "test_file.txt"

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
        tmp_file.write(TEST_CONTENT.encode('utf-8'))
        mocked_file_path = tmp_file.name

    handler = AWSHandler('mocked_bucket-user-role')
    handler.upload_file_to_s3(mocked_bucket_name, mocked_file_path, TEST_FILE_NAME)
    os.remove(mocked_file_path)

    mocked_s3_resource = boto3.resource('s3', region_name = mocked_region)
    object = mocked_s3_resource.Object(mocked_bucket_name, TEST_FILE_NAME)
    file_content = object.get()['Body'].read().decode('utf-8')

    assert file_content == TEST_CONTENT

@mock_aws
@patch.dict(os.environ, {
    'AWS_ACCESS_KEY_ID': 'access_key',
    'AWS_SECRET_ACCESS_KEY': 'secret_key',
    'ACCOUNT_ID': '123456789012'
})
def test_upload_buffer_to_s3():
    """
    Tests the upload_buffer_to_s3 method of AWSHandler.

    Verifies that method successfully uploads content from a buffer to an S3 bucket.
    It mocks AWS S3 to create a bucket and handle the buffer upload. The test
    creates a StringIO buffer with test content and passes it to the handler.

    Asserts:
        The content of the uploaded file in the S3 bucket matches the content from the buffer.
    """

    mocked_bucket_name = 'mocked-bucket'
    mocked_region = 'us-east-1'
    mocked_s3_client = boto3.client('s3', region_name = mocked_region)
    mocked_s3_client.create_bucket(Bucket = mocked_bucket_name)

    buffer_content = io.StringIO(TEST_CONTENT)
    handler = AWSHandler('mocked_bucket-user-role')
    handler.upload_buffer_to_s3(mocked_bucket_name, buffer_content, TEST_FILE_NAME)

    mocked_s3_resource = boto3.resource('s3', region_name = mocked_region)
    object = mocked_s3_resource.Object(mocked_bucket_name, TEST_FILE_NAME)
    file_content = object.get()['Body'].read().decode('utf-8')

    assert file_content == TEST_CONTENT

@mock_aws
@patch.dict(os.environ, {
    'AWS_ACCESS_KEY_ID': 'access_key',
    'AWS_SECRET_ACCESS_KEY': 'secret_key',
    'ACCOUNT_ID': '123456789012'
})
def test_download_file_from_s3():
    """
    Tests the download_file_from_s3 method of AWSHandler.

    Verifies that method successfully downloads a file from an S3 bucket.
    It mocks AWS S3 to create a bucket, uploads test content, and then
    downloads the file.

    Asserts:
        The content of the downloaded file matches what was originally uploaded.
    """

    mocked_bucket_name = 'mocked-bucket'
    mocked_region = 'us-east-1'
    mocked_s3_client = boto3.client('s3', region_name = mocked_region)
    mocked_s3_client.create_bucket(Bucket = mocked_bucket_name)

    mocked_s3_client.put_object(
        Bucket = mocked_bucket_name,
        Key = TEST_FILE_NAME,
        Body = TEST_CONTENT
    )

    with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
        download_path = tmp_file.name

    handler = AWSHandler('mocked_bucket-user-role')
    handler.download_file_from_s3(mocked_bucket_name, TEST_FILE_NAME, download_path)

    with open(download_path, 'r') as file:
        downloaded_content = file.read()

    os.remove(download_path)

    assert downloaded_content == TEST_CONTENT