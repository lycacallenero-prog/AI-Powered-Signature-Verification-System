import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('Starting Signature Verification Backend...');

// Change to backend directory
process.chdir(join(__dirname, 'backend'));

console.log('Note: This environment has limited Python support.');
console.log('The backend requires TensorFlow and OpenCV which are not available in WebContainer.');
console.log('Please run this project in a local environment with full Python support.');
console.log('');
console.log('To run locally:');
console.log('1. Install Python 3.8+');
console.log('2. Run: pip install -r backend/requirements.txt');
console.log('3. Run: cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload');

process.exit(1);