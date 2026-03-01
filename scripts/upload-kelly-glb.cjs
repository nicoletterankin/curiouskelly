const { put } = require('@vercel/blob');
const fs = require('fs');
const path = require('path');

const BLOB_TOKEN = 'vercel_blob_rw_z4yuma7KJ5h9tD7V_OxHdaIhoBmRVS5BxfJrqwzFWXcyCie';
const FILE_PATH = 'C:\\Users\\user\\OneDrive\\Desktop\\kelly-final.glb';

async function upload() {
  console.log('Reading file...');
  const fileBuffer = fs.readFileSync(FILE_PATH);
  const sizeMB = (fileBuffer.length / (1024 * 1024)).toFixed(2);
  console.log(`File size: ${sizeMB} MB`);

  console.log('Uploading to Vercel Blob...');
  const blob = await put('models/kelly.glb', fileBuffer, {
    access: 'public',
    token: BLOB_TOKEN,
    contentType: 'model/gltf-binary',
    addRandomSuffix: false,
  });

  console.log('Upload complete!');
  console.log('URL:', blob.url);
  console.log('Pathname:', blob.pathname);
}

upload().catch(err => {
  console.error('Upload failed:', err.message);
  process.exit(1);
});
