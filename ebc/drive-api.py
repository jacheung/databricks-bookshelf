{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "9f515062-41a2-4114-8935-5f2aa4326020",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "!pip install google-api-python-client"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "ececd75a-b8f3-4a3c-9ac2-43a5de8064b2",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "from google.oauth2 import service_account\n",
    "from googleapiclient.discovery import build\n",
    "import os\n",
    "\n",
    "SCOPES = ['https://www.googleapis.com/auth/drive']\n",
    "SERVICE_ACCOUNT_FILE = \"/Volumes/main/jon_cheung/gcp-svc-credentials/gcp-svc-credentials.json\"\n",
    "\n",
    "credentials = service_account.Credentials.from_service_account_file(\n",
    "    SERVICE_ACCOUNT_FILE, scopes=SCOPES)\n",
    "\n",
    "service = build('drive', 'v3', credentials=credentials)\n",
    "\n",
    "# Example: List the first 10 files in Google Drive\n",
    "results = service.files().list(pageSize=10).execute()\n",
    "items = results.get('files', [])\n",
    "\n",
    "if not items:\n",
    "    print('No files found.')\n",
    "else:\n",
    "    print('Files:')\n",
    "    for item in items:\n",
    "        print(u'{0} ({1})'.format(item['name'], item['id']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "16b847b5-57e4-4b3a-a568-beeb8562202f",
     "showTitle": false,
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "service.files().list(supportsAllDrives=True, includeItemsFromAllDrives=True).execute()\n"
   ]
  }
 ],
 "metadata": {
  "application/vnd.databricks.v1+notebook": {
   "dashboards": [],
   "environmentMetadata": {
    "base_environment": "",
    "client": "1"
   },
   "language": "python",
   "notebookMetadata": {
    "pythonIndentUnit": 2
   },
   "notebookName": "drive-api",
   "widgets": {}
  },
  "kernelspec": {
   "display_name": "sandbox",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
