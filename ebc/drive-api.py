{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "#encode base64 text file\n",
    "import base64\n",
    "import json\n",
    "\n",
    "# open up the credentials json and encode it using base64\n",
    "credentials = \"/Users/jon.cheung/GitHub/credentials/gcp-svc-credentials.json\"\n",
    "with open(credentials) as f:\n",
    "    dict = json.load(f)\n",
    "    string = json.dumps(dict)\n",
    "credentials_b64_bytes = base64.b64encode(string.encode('utf-8'))\n",
    "credentials_b64_string = credentials_b64_bytes.decode(\"utf-8\")\n",
    "\n",
    "# when using databricks secrets CLI, the payload must contain the following keys: 'scope', 'key', 'string_value'\n",
    "databricks_secrets_json = {'scope': 'ebc-application',\n",
    "                           'key': 'gcp-service-account-for-gdrive',\n",
    "                           'string_value': credentials_b64_string}\n",
    "j = json.dumps(databricks_secrets_json)\n",
    "with open('databricks-secrets-gcp-svc-payload.json', 'w') as f:\n",
    "    print(j, file=f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'type': 'service_account',\n",
       " 'project_id': 'cedar-binder-385618',\n",
       " 'private_key_id': '55f459985df0335d20572f3a8f6d32993ce8c082',\n",
       " 'private_key': '-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC9OQeMfxBGhGlx\\n22Fo8jAZGTHgKmTAbiOPXGXKXdJDBel8s5vkCiGd3pYSrUtbsrgwmf0k+PRglCtj\\noB6k67rPFIItfhsSH8jOPEr8XSNfk5HIdSFruWVx1fWc0YC5KJOk0xdTcR5Frfzc\\n0UM+vLnoI+Qa7w4lxCaWboH/jErVy30Z7mgCl2oi9cwbUic5a3cyWM7J56s035SM\\nVL40uPy/91m9yI78SYmO5Zzi3FB+FgaeBTLaNDEqe23pEcgla5Qg0eVpY9jpHyLz\\np1EyrQpV71GAxtjFjLHohK2uIFqdtMLGvHyt21PUUjx1PMi2ZuOnh950gry5J9Sd\\nhwLVcAuxAgMBAAECggEABoMVn1xBW7Lrg9SJanygwwFc5/5/gYec086JzoRKc93q\\niF0QS7GdUO7/zlBeVj9dI0eOaKkEdIUu5dalLVbzQrSNixoinX3gadgVetBMAwpb\\np1jgG8Jpc0iqM3Galuy2gZL189xCDuL3L4SIZlqlgX+oNlTVMV+p9g4DLW+3JNrE\\n5gwu50inaTYrEHxXlK73X0LrcmsdLGRem8eBf8ofrdrFjSbkLHC5vACM/wktBDDT\\n5409bc4L0ib3Y9dV1CeXCM8XDUlPzyTCyjCM5mlq710gxPhaV3NRRoGulbwKYMqv\\ndcOEqeiSFUm1N6HxuO+Pd0SgY2/nWhyDR9VE6b6LQQKBgQD0OGsksQioVrX1FPgg\\nbmrZ5nMG6A8YasOwae4xsoPYkdM05cHjibhTtitv9eHDcsOWW7e3ZOfgIdmet0B4\\naAPc6nD2iw3r27TtCBVxW8R2dfcZef+PQ+Y47Snl0+nPmW2y0SJeqvYaffdddDLK\\nwTGsb93zpgTU2C6vMXmBSx90cQKBgQDGWYMZJ+GUCSYxOnl84iM4N5GQ4TjOjGL1\\nXOfeX/yMmUtHH5xWNhpEabz2rrd6iXpA1AZU+4ag/RQwDJHr+oHy8N88VZ87NjJ+\\nvQB6/ST29bg692f5AVEJG9JKGZrNsM9StIgSQgFPWDy/4TCoCs/3sjysc1RdvHum\\n54vFoGyrQQKBgQDBOzqbJ2/gMet6ZQMGNhdZHVt55Xa8LQ10sfwDWmmzm33vZrMI\\nY9lycrrftT7SNCXI+/zaoH5O+rmDOR9LpZEY5G5IxDFZotb+jNzaem9yA1hl0f+A\\nzYqFFDGIZSmkVpNTXuHA4agjwfNNADmH72BsPX7x3zQHbJ1ThOPMhzH2IQKBgBaJ\\nGQIEq/Z7y6EQOblcoA/FEH7bd+7PuHaUJav4T+NRj2H3T3XkE7vuH5APbb04XXF8\\nJXQGV9d2qZKD+xhKj/UgNNzQBZVepQINSz6uAEMmy9W3QlLiOWjFhnDw08vG6OdG\\np6cbZLa1GcHbPgH5qINF9urI967muU3PROwhO/eBAoGBAM5WU5JJOLG7YGMH0982\\nGN6u4yEBYLbQFEIOCHeqVwqPXBsswdocN9Mn9JxGx+cAqfunmMZgfEWz+VYzAb0O\\nHU9VkvBTZJQgEnw6Vg+LRYrzsTIoRqEzZunt4cWSHF5eiDwXfrKf70uIoEDsmTk/\\nYzKdu2pip/Z6IPxVctPBv5gO\\n-----END PRIVATE KEY-----\\n',\n",
       " 'client_email': 'svc-ebc-app@cedar-binder-385618.iam.gserviceaccount.com',\n",
       " 'client_id': '104565612682517113382',\n",
       " 'auth_uri': 'https://accounts.google.com/o/oauth2/auth',\n",
       " 'token_uri': 'https://oauth2.googleapis.com/token',\n",
       " 'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',\n",
       " 'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/svc-ebc-app%40cedar-binder-385618.iam.gserviceaccount.com',\n",
       " 'universe_domain': 'googleapis.com'}"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "b64_bytes = credentials_b64_string.encode('utf-8')\n",
    "credentials_bytes = base64.b64decode(b64_bytes)\n",
    "credentials_string = credentials_bytes.decode(\"utf-8\")\n",
    "credentials = json.loads(credentials_string)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No files found.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "from google.oauth2 import service_account\n",
    "from googleapiclient.discovery import build\n",
    "import os\n",
    "\n",
    "SCOPES = ['https://www.googleapis.com/auth/drive']\n",
    "SERVICE_ACCOUNT_FILE = \"/Users/jon.cheung/GitHub/credentials/gcp-svc-credentials.json\"\n",
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
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'kind': 'drive#fileList', 'incompleteSearch': False, 'files': []}"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "service.files().list(supportsAllDrives=True, includeItemsFromAllDrives=True).execute()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
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
 "nbformat_minor": 2
}
