import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import suggestionsRouter from './routes/suggestions.js';
import logsRouter from './routes/logs.js';
import testRouter from './routes/test.js';

dotenv.config();

console.log('Server starting on port', process.env.PORT || 3001);
console.log('CORS Origins:', ['https://f3273c13.mydala.app']);

const app = express();
const PORT = process.env.PORT || 3001;

// Allow both default Vite port and the one specified in package.json
app.use(cors({
  origin: ['https://f3273c13.mydala.app', 'http://localhost:5173', 'http://localhost:3000'],
  methods: ['GET', 'POST']
}));
app.use(express.json());

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use('/api/suggestions', suggestionsRouter);
app.use('/api/logs', logsRouter);
app.use('/api/test', testRouter);

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});
