import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Load environment variables from the .env file
load_dotenv()

def test_huggingface_connection():
    # Retrieve the API key using the correct variable name matching your .env file
    api_key = os.getenv("HF_TOKEN")
    
    if not api_key:
        print("Error: HUGGINGFACEHUB_API_TOKEN environment variable not found.")
        print("Please ensure your .env file exists and contains this variable.")
        return

    try:
        print("Attempting to authenticate with Hugging Face...")
        login(token=api_key, add_to_git_credential=False)
        
        api = HfApi()
        user_info = api.whoami()
        username = user_info.get("name")
        
        print("\n--- Connection Successful! ---")
        print(f"Authenticated successfully as user: {username}")
        
    except Exception as e:
        print("\n--- Connection Failed ---")
        print(f"An error occurred while connecting to Hugging Face: {e}")

if __name__ == "__main__":
    test_huggingface_connection()