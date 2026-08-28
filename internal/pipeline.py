import re
from pathlib import Path
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = BASE_DIR / "internal" / "service_account.json"
OUTPUT_FILE = BASE_DIR / "data" / "master_clean.csv"

# Make sure this matches your exact spreadsheet name
SHEET_NAME = "Apple Device Trade-In Value Prediction & Market Analysis" 

# Updated to match the real columns in your Google Sheet
REQUIRED_COLUMNS = [
    "Device", 
    "Standardized Model", 
    "Storage (GB)", 
    "Max. Trade-In Value (RM)"
]

def get_sheet_data() -> pd.DataFrame:
    print("Authenticating with Google Sheets...")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    
    print("Downloading records...")
    # NOTE: Change "Sheet2" to your exact tab name if needed
    sheet = client.open(SHEET_NAME).worksheet("Cleaned Data (ML use)") 
    
    records = sheet.get_all_records()
    return pd.DataFrame(records)

def parse_storage(val):
    """Converts '128 GB' -> 128.0 and '1 TB' -> 1024.0"""
    val = str(val).upper().strip()
    
    # Extract the number from the string
    num_match = re.search(r'([\d.]+)', val)
    if not num_match:
        return pd.NA
        
    num = float(num_match.group(1))
    
    # Convert Terabytes to Gigabytes if necessary
    if 'TB' in val:
        num *= 1024
        
    return num

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Initial row count: {len(df)}")
    
    # 1. Column verification
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sheet: {missing}")

    # 2. String standardization
    df["Device"] = df["Device"].astype(str).str.strip()
    df["Standardized Model"] = df["Standardized Model"].astype(str).str.strip()
    
    # 3. Clean Storage Column (Strip text and convert to numeric)
    df["Storage (GB)"] = df["Storage (GB)"].apply(parse_storage)
    df["Storage (GB)"] = pd.to_numeric(df["Storage (GB)"], errors="coerce")
    
    # 4. Clean Trade-In Value (Ensure it's a number)
    df["Max. Trade-In Value (RM)"] = pd.to_numeric(df["Max. Trade-In Value (RM)"], errors="coerce")
    
    # 5. Filter invalid rows (Drop empty models or zero/missing prices)
    clean_df = df[
        (df["Device"] != "") &
        (df["Standardized Model"] != "") &
        (df["Max. Trade-In Value (RM)"] > 0)
    ].copy()

    # Drop duplicate records
    clean_df.drop_duplicates(inplace=True)
    
    print(f"Cleaned row count: {len(clean_df)}")
    return clean_df

if __name__ == "__main__":
    raw_df = get_sheet_data()
    clean_df = validate_and_clean(raw_df)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! {len(clean_df)} validated records written to {OUTPUT_FILE}.")