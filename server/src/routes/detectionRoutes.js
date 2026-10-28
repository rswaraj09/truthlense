import express from 'express';
import {
  detectImage,
  detectVideo,
  getDetectionHistory,
  getDetectionById,
} from '../controllers/detectionController.js';
import { optionalAuth, requireAuth } from '../middleware/authMiddleware.js';
import { uploadImage, uploadVideo } from '../middleware/uploadMiddleware.js';

const router = express.Router();

// Detection endpoints (accessible to guests and authenticated users)
router.post('/image', optionalAuth, uploadImage.single('file'), detectImage);
router.post('/video', optionalAuth, uploadVideo.single('file'), detectVideo);

// History and specific detection retrieval
router.get('/history', requireAuth, getDetectionHistory);
router.get('/:id', optionalAuth, getDetectionById);

export default router;
