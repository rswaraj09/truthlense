import dotenv from 'dotenv';
dotenv.config();

import app from './app.js';
import { connectDB } from './config/db.js';

const PORT = process.env.PORT || 5000;

// Connect to MongoDB
connectDB();

const server = app.listen(PORT, () => {
  console.log(`===============================================`);
  console.log(`[TruthLens API] Server running on port ${PORT}`);
  console.log(`[TruthLens API] Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`[TruthLens API] AI Service URL: ${process.env.AI_SERVICE_URL || 'http://127.0.0.1:8000'}`);
  console.log(`===============================================`);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('[TruthLens API] SIGTERM received. Shutting down gracefully...');
  server.close(() => {
    console.log('[TruthLens API] Process terminated');
  });
});
