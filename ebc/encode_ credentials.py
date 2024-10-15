# Databricks notebook source
#encode base64 text file
import base64
import json

# open up the credentials json and encode it using base64
credentials = "/Users/jon.cheung/GitHub/credentials/gcp-svc-credentials.json"
with open(credentials) as f:
    dict = json.load(f)
    string = json.dumps(dict)
credentials_b64_bytes = base64.b64encode(string.encode('utf-8'))
credentials_b64_string = credentials_b64_bytes.decode("utf-8")

# when using databricks secrets CLI, the payload must contain the following keys: 'scope', 'key', 'string_value'
databricks_secrets_json = {'scope': 'ebc-application',
                           'key': 'gcp-service-account-for-gdrive',
                           'string_value': credentials_b64_string}
j = json.dumps(databricks_secrets_json)
with open('databricks-secrets-gcp-svc-payload.json', 'w') as f:
    print(j, file=f)

# COMMAND ----------

# decode base64 credentials
import base64
import json

# view secrets within secret scope
dbutils.secrets.list('ebc-application')
credentials_b64_string = dbutils.secrets.get('ebc-application', 'gcp-service-account-for-gdrive')

# decode credentials from base64
b64_bytes = credentials_b64_string.encode('utf-8')
credentials_bytes = base64.b64decode(b64_bytes)
credentials_string = credentials_bytes.decode("utf-8")
credentials = json.loads(credentials_string)
