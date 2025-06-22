# -*- coding: utf-8 -*-
"""Provides utility for accessing secrets from Google Cloud Secret Manager."""

from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """
    Access the payload of the given secret version.

    Args:
        project_id: Google Cloud project ID.
        secret_id: ID of the secret.
        version_id: ID of the secret version (or "latest").

    Returns:
        The secret payload as a string.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

if __name__ == '__main__':
    # This is an example of how to use this function.
    # You would need to set your GOOGLE_APPLICATION_CREDENTIALS environment
    # variable for this to work, or be running in an environment with
    # appropriate default credentials.
    # import os
    # current_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    # if current_project_id:
    #     # Replace 'your-secret-id' with an actual secret ID in your project
    #     try:
    #         retrieved_secret = get_secret(current_project_id, "your-secret-id")
    #         print(f"Successfully retrieved secret: {retrieved_secret[:10]}...") # Print first 10 chars
    #     except Exception as e:
    #         print(f"Error retrieving secret: {e}")
    # else:
    #     print("GOOGLE_CLOUD_PROJECT environment variable not set. Cannot run example.")
    pass
