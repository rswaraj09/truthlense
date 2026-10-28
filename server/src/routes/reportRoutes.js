import express from 'express';
import {
  createReport,
  getReportByRef,
  getUserReports,
} from '../controllers/reportController.js';
import { optionalAuth, requireAuth } from '../middleware/authMiddleware.js';

const router = express.Router();

// Report submission (authenticated or guest with contact info)
router.post('/', optionalAuth, createReport);

// View user's own reports
router.get('/my-reports', requireAuth, getUserReports);

// Public/submitter status tracking by reference number
router.get('/:ref', optionalAuth, getReportByRef);

export default router;
