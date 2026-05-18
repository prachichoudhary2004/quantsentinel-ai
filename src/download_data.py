import os
import requests

def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive using its file ID, handling large file confirmations.
    """
    print(f"Initiating download for file ID: {file_id}...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # First request to get the download page / check for confirmation token
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        print("Confirmation token found. Proceeding with download...")
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    else:
        print("No confirmation token needed. Downloading directly...")
        
    save_response_content(response, destination)
    print(f"Finished downloading. Saved to: {destination}")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    # Google Drive File IDs from the links provided
    # Historical Trader Data: https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing
    historical_data_id = "1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs"
    # Fear & Greed Index: https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing
    fear_greed_id = "1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf"
    
    # Save destinations (we'll start by assuming they are CSV files and inspect them)
    # We will save them with raw names and inspect their content
    os.makedirs("data/raw", exist_ok=True)
    
    download_file_from_google_drive(historical_data_id, "data/raw/hyperliquid_trader_data.csv")
    download_file_from_google_drive(fear_greed_id, "data/raw/bitcoin_fear_greed_index.csv")
