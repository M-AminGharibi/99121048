import os
import random
import requests
from bs4 import BeautifulSoup

# List of animal types to search for
animal_types = ['whale', 'sparrow']

# Create a directory to store the downloaded images
os.makedirs('animal_images/whale', exist_ok=True)
os.makedirs('animal_images/sparrow', exist_ok=True)

# Base URL for Google image search
GOOGLE_IMAGE = 'https://www.google.com/search?site=&source=hp&biw=1873&bih=990&tbm=isch&q='

for animal_type in animal_types:
    print(f"Searching for images of {animal_type}...")

    # Construct the search URL
    search_url = GOOGLE_IMAGE + animal_type

    # Send a GET request to the search URL
    response = requests.get(search_url)

    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all <img> tags
    img_tags = soup.find_all('img')

    # Download and save the images
    for i, img_tag in enumerate(img_tags):
        try:
            # Get the image URL
            img_url = img_tag['src']

            # Send a GET request to the image URL
            img_data = requests.get(img_url).content

            # Generate a random filename for the image
            filename = f"{animal_type}_{random.randint(1, 1000000)}.jpg"

            # Save the image to the appropriate directory
            with open(f"animal_images/{animal_type}/{filename}", 'wb') as f:
                f.write(img_data)

            print(f"Downloaded {filename}")

            # Limit the number of images downloaded per animal type
            if i >= 150:
                break
        except Exception as e:
            print(f"Could not download image: {e}")
            continue

print("Image download completed.")
