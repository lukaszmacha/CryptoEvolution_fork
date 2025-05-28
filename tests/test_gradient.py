# tests/test_gradient.py

from source.gradient import GradientHandler, DEFAULT_START_COMMAND
from unittest.mock import Mock, ANY, patch
import os

@patch('gradient.NotebooksClient.create', new_callable=Mock)
@patch.dict(os.environ, {
    'GRADIENT_API_KEY': 'api_key',
    'GRADIENT_PROJECT_ID': 'project_id'
})
def test_created_notebook__with_2_machine_types_1_fail_1_success(mock_notebooks_client_create):
    """
    Tests the create_notebook method of GradientHandler.

    Verifies that the create_notebook method successfully creates a notebook using 
    the Paperspace Gradient API. The NotebooksClient's create method is mocked to simulate the creation 
    process without making actual API calls.

    Asserts:
        - The notebook ID returned by create_notebook matches the mocked notebook ID.
        - The NotebooksClient's create method was called once with the correct parameters.
    """

    mocked_error = Exception("""Failed to create resource: We are currently out of capacity for the
                                selected VM type. Try again in a few minutes, or select a different instance.""")
    mocked_notebook_id = 'NOTEBOOK_ID'
    mock_notebooks_client_create.side_effect = [mocked_error, mocked_notebook_id]

    expected_workspace = 'https://github.com/user/repo_name.git'
    expected_machine_types = ['Free-A4000', 'Free-P5000']
    expected_container = 'paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125'
    expected_timeout = 6
    expected_command = 'echo "Testing command"'
    expected_command_full = expected_command + DEFAULT_START_COMMAND
    expected_environment = dict()

    gradient_handler = GradientHandler()
    notebook_id = gradient_handler.create_notebook(github_repository_url = expected_workspace, 
                                                   command_to_invoke = expected_command,
                                                   machine_types = expected_machine_types)
    
    assert notebook_id == mocked_notebook_id
    assert mock_notebooks_client_create.call_count == len(expected_machine_types)
    for expected_machine_type in expected_machine_types:
        mock_notebooks_client_create.assert_any_call(machine_type = expected_machine_type,
                                                     container = expected_container,
                                                     project_id = gradient_handler.project_id,
                                                     shutdown_timeout = expected_timeout,
                                                     workspace = expected_workspace,
                                                     command = expected_command_full,
                                                     environment = expected_environment,
                                                     name = ANY)