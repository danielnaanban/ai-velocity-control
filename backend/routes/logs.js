import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const router = express.Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

router.get('/', (req, res) => {
    const fillsPath = path.join(__dirname, '..', 'data', 'fills.csv');
    
    try {
        if (!fs.existsSync(fillsPath)) {
            return res.json([]);
        }
        
        const data = fs.readFileSync(fillsPath, 'utf8');
        const lines = data.split('
').filter(line => line.trim() !== '');
        
        // Skip header
        const logs = lines.slice(1).map(line => {
            const [timestamp, pair, action, entry, sl, tp, confidence] = line.split(',');
            return `[EXECUTION] ${action} ${pair} @ ${entry} (Conf: ${confidence})`;
        });
        
        // Return last 20 logs
        res.json(logs.slice(-20));
    } catch (err) {
        console.error('Logs fetch error:', err);
        res.status(500).json({ error: 'Failed to fetch logs' });
    }
});

export default router;
