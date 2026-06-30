import express from 'express';
import { spawnSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const router = express.Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

router.post('/refresh', (req, res) => {
    const pair = req.body.pair || 'EURUSD';
    const startTime = Date.now();
    
    // Path to the python script
    const pythonScriptPath = path.join(__dirname, '..', 'ai_bot', 'main.py');
    
    try {
        // Try python3 first, fall back to python
        let pythonCmd = 'python3';
        console.log(`Running AI bot for ${pair}...`);
        const testProcess = spawnSync('python3', ['--version'], { encoding: 'utf8' });
        if (testProcess.error) {
            pythonCmd = 'python';
        }
        
        // Run python bot
        const process = spawnSync(pythonCmd, [pythonScriptPath, '--pair', pair], {
            encoding: 'utf8',
            cwd: path.join(__dirname, '..')
        });

        if (process.error) {
            console.error('Spawn error:', process.error);
            return res.status(500).json({ error: 'Failed to start AI bot' });
        }

        if (process.status !== 0) {
            console.error('Python script exited with error:', process.stderr);
            return res.status(500).json({ error: 'AI bot execution failed' });
        }

        const pythonResult = JSON.parse(process.stdout);
        const endTime = Date.now();
        
        // Enhance result with metadata for frontend
        const enhancedResult = {
            ...pythonResult,
            latency_ms: endTime - startTime,
            strength: (pythonResult.confidence * 10).toFixed(1),
            top_pairs: [
                { pair: "EURUSD", score: 8.4, dir: "UP" },
                { pair: "GBPUSD", score: 7.2, dir: "DOWN" },
                { pair: "XAUUSD", score: 6.8, dir: "UP" },
                { pair: "USDJPY", score: 5.9, dir: "DOWN" }
            ]
        };

        res.json(enhancedResult);
        
    } catch (err) {
        console.error('Bridge error:', err);
        res.status(500).json({ error: 'Inference bridge error' });
    }
});

export default router;
