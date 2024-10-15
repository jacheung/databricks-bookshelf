# Databricks notebook source
!pip install google-api-python-client

# COMMAND ----------


from google.oauth2 import service_account
from googleapiclient.discovery import build
import os

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = "/Volumes/main/jon_cheung/gcp-svc-credentials/gcp-svc-credentials.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('drive', 'v3', credentials=credentials)

# Example: List the first 10 files in Google Drive
results = service.files().list(pageSize=10).execute()
items = results.get('files', [])

if not items:
    print('No files found.')
else:
    print('Files:')
    for item in items:
        print(u'{0} ({1})'.format(item['name'], item['id']))

# COMMAND ----------

service.files().list(supportsAllDrives=True, includeItemsFromAllDrives=True).execute()

