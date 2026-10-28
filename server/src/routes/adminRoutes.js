import express from 'express';
import {
  getAllReports,
  updateReportStatus,
  updateRemovalRequest,
} from '../controllers/reportController.js';
import { requireAuth } from '../middleware/authMiddleware.js';
import { requireAdmin } from '../middleware/adminMiddleware.js';

const router = express.Router();

// All administrative routes require valid JWT and admin role
router.use(requireAuth, requireAdmin);

router.get('/reports', getAllReports);
router.patch('/reports/:id/status', updateReportStatus);
router.post('/reports/:id/removal-request', updateRemovalRequest);

export default router;
