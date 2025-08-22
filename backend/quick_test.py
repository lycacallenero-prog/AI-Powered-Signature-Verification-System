import requests
import json
import io
import numpy as np
from PIL import Image, ImageDraw

def create_simple_signature(width=200, height=100, seed=42):
    """Create a simple test signature"""
    np.random.seed(seed)
    
    # Create white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple signature curve
    points = []
    for x in range(20, width-20, 10):
        y = height//2 + int(15 * np.sin(x/30)) + np.random.randint(-5, 5)
        points.append((x, y))
    
    # Draw the signature
    for i in range(len(points)-1):
        draw.line([points[i], points[i+1]], fill='black', width=3)
    
    return img

def test_training():
    """Test the training endpoint"""
    print("Creating test signatures...")
    
    # Create 5 test signatures
    files = []
    for i in range(5):
        sig_img = create_simple_signature(seed=i*123)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        sig_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files.append(('training_images', (f'sig_{i}.png', img_bytes.getvalue(), 'image/png')))
    
    print("Sending training request...")
    try:
        response = requests.post(
            'http://localhost:8000/train',
            files=files,
            timeout=60  # 1 minute timeout
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Training request successful!")
            print("Response content (first 500 chars):")
            print(response.text[:500])
        else:
            print(f"❌ Training failed: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ Training request timed out (this is expected for actual training)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test health first
    try:
        health = requests.get('http://localhost:8000/health', timeout=5)
        print(f"Health check: {health.json()}")
        
        if health.status_code == 200:
            test_training()
        else:
            print("Server not healthy")
    except Exception as e:
        print(f"Server not accessible: {e}")