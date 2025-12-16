import firebase_admin
from firebase_admin import credentials, auth

# Path to your service account key
cred = credentials.Certificate("cropsense-firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# 🔴 PASTE THE ADMIN UID HERE
ADMIN_UID = "9ht9VUQ1kWf53DVgbh80JuCZxrU2"

auth.set_custom_user_claims(ADMIN_UID, {"role": "admin"})

print("✅ Admin role successfully assigned")
