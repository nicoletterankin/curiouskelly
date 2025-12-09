/**
 * Kelly Asset Review Tool - Server
 * 
 * Serves the review UI and handles file operations (approve/reject).
 * Usage: node scripts/kelly-visual-identity/review-server.js
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static('public')); // Serve images

// Paths
const BASE_DIR = path.join(process.cwd(), 'public', 'kelly', 'lessons');
const APPROVED_DIR = path.join(process.cwd(), 'public', 'kelly', 'approved');
const REJECTED_DIR = path.join(process.cwd(), 'public', 'kelly', 'rejected');

// Ensure dirs exist
[APPROVED_DIR, REJECTED_DIR].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// API: Get list of images to review
app.get('/api/images', (req, res) => {
    const images = [];
    
    function scanDir(directory) {
        const files = fs.readdirSync(directory);
        for (const file of files) {
            const fullPath = path.join(directory, file);
            if (fs.statSync(fullPath).isDirectory()) {
                scanDir(fullPath);
            } else if (file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.webp')) {
                // Return relative path from 'public'
                const relPath = path.relative(path.join(process.cwd(), 'public'), fullPath);
                images.push(relPath);
            }
        }
    }
    
    if (fs.existsSync(BASE_DIR)) {
        scanDir(BASE_DIR);
    }
    
    res.json(images);
});

// API: Move file
app.post('/api/move', (req, res) => {
    const { filePath, action } = req.body; // filePath is relative to 'public'
    
    if (!filePath || !action) return res.status(400).send("Missing data");
    
    const sourcePath = path.join(process.cwd(), 'public', filePath);
    const fileName = path.basename(filePath);
    const targetDir = action === 'approve' ? APPROVED_DIR : REJECTED_DIR;
    const targetPath = path.join(targetDir, fileName);
    
    try {
        if (fs.existsSync(sourcePath)) {
            fs.renameSync(sourcePath, targetPath);
            console.log(`Moved [${action}]: ${fileName}`);
            res.json({ success: true });
        } else {
            res.status(404).send("File not found");
        }
    } catch (e) {
        console.error(e);
        res.status(500).send(e.message);
    }
});

// Serve the UI
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'review-client.html'));
});

app.listen(PORT, () => {
    console.log(`👀 Review Tool running at http://localhost:${PORT}`);
});






