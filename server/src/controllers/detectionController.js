import mongoose from 'mongoose';
import { Detection } from '../models/Detection.js';
import { AIService } from '../services/aiService.js';
import { validateFileSignature, cleanupFile } from '../middleware/uploadMiddleware.js';

// Save the result when MongoDB is connected; otherwise keep detection working
// (the record just isn't stored), so the demo never depends on the database.
const persist = async (doc) => {
  if (mongoose.connection.readyState !== 1) {
    return { ...doc, _id: null, createdAt: new Date() };
  }
  return Detection.create(doc);
};

export const detectImage = async (req, res, next) => {
  const file = req.file;
  if (!file) {
    return res.status(400).json({
      success: false,
      message: 'No image file uploaded. Please provide a JPG, PNG, or WEBP image.',
    });
  }

  try {
    // 1. Validate actual file signature (magic bytes)
    const isValidSignature = validateFileSignature(file.path, 'image');
    if (!isValidSignature) {
      return res.status(400).json({
        success: false,
        message: 'Invalid file content: The uploaded file does not match a valid image signature.',
      });
    }

    // 2. Call Python AI service for inference
    const aiResult = await AIService.predictImage(file.path, file.originalname);

    if (!aiResult.success) {
      return res.status(502).json({
        success: false,
        message: aiResult.error || 'AI inference service returned an error',
        details: aiResult,
      });
    }

    // 3. Save detection record to MongoDB
    const detectionRecord = await persist({
      userId: req.user ? req.user._id : null,
      mediaType: 'image',
      originalFilename: file.originalname,
      fileSize: file.size,
      mimeType: file.mimetype,
      prediction: aiResult.prediction,
      label: aiResult.label,
      modelScore: aiResult.score,
      confidenceLevel: aiResult.confidenceLevel || 'medium',
      processingTimeMs: aiResult.processingTimeMs || 0,
      isPreliminaryScreening: false,
      forensicMetrics: aiResult.forensicMetrics || {},
      limitationDisclaimer:
        aiResult.limitation ||
        'Model predictions are statistical probabilistic estimates and should not be used as sole legal proof of authenticity.',
    });

    // 4. Return response to client
    res.status(200).json({
      success: true,
      detectionId: detectionRecord._id,
      prediction: detectionRecord.prediction,
      label: detectionRecord.label,
      score: detectionRecord.modelScore,
      confidenceLevel: detectionRecord.confidenceLevel,
      processingTimeMs: detectionRecord.processingTimeMs,
      forensicMetrics: detectionRecord.forensicMetrics,
      limitationDisclaimer: detectionRecord.limitationDisclaimer,
      createdAt: detectionRecord.createdAt,
    });
  } catch (err) {
    next(err);
  } finally {
    // Always clean up uploaded temporary file
    cleanupFile(file?.path);
  }
};

export const detectVideo = async (req, res, next) => {
  const file = req.file;
  if (!file) {
    return res.status(400).json({
      success: false,
      message: 'No video file uploaded. Please provide an MP4, WebM, AVI, or MOV video.',
    });
  }

  try {
    // 1. Validate actual file signature (magic bytes)
    const isValidSignature = validateFileSignature(file.path, 'video');
    if (!isValidSignature) {
      return res.status(400).json({
        success: false,
        message: 'Invalid file content: The uploaded file does not match a valid video signature.',
      });
    }

    // 2. Call Python AI service for frame screening
    const aiResult = await AIService.predictVideo(file.path, file.originalname);

    if (!aiResult.success) {
      return res.status(502).json({
        success: false,
        message: aiResult.error || 'AI video screening returned an error',
        details: aiResult,
      });
    }

    // 3. Save detection record to MongoDB
    const detectionRecord = await persist({
      userId: req.user ? req.user._id : null,
      mediaType: 'video',
      originalFilename: file.originalname,
      fileSize: file.size,
      mimeType: file.mimetype,
      prediction: aiResult.prediction,
      label: aiResult.label,
      modelScore: aiResult.score,
      confidenceLevel: aiResult.confidenceLevel || 'medium',
      processingTimeMs: aiResult.processingTimeMs || 0,
      isPreliminaryScreening: true,
      forensicMetrics: aiResult.forensicMetrics || {},
      framesAnalysis: aiResult.frames || [],
      limitationDisclaimer:
        aiResult.limitation ||
        'PRELIMINARY FRAME-LEVEL SCREENING: This model classifies individual static sampled frames and does not model temporal or deepfake audio/motion consistency. Results must be interpreted with caution.',
    });

    // 4. Return response to client
    res.status(200).json({
      success: true,
      detectionId: detectionRecord._id,
      prediction: detectionRecord.prediction,
      label: detectionRecord.label,
      score: detectionRecord.modelScore,
      confidenceLevel: detectionRecord.confidenceLevel,
      processingTimeMs: detectionRecord.processingTimeMs,
      isPreliminaryScreening: true,
      forensicMetrics: detectionRecord.forensicMetrics,
      framesAnalysis: detectionRecord.framesAnalysis,
      limitationDisclaimer: detectionRecord.limitationDisclaimer,
      createdAt: detectionRecord.createdAt,
    });
  } catch (err) {
    next(err);
  } finally {
    // Always clean up uploaded temporary file
    cleanupFile(file?.path);
  }
};

export const getDetectionHistory = async (req, res, next) => {
  try {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required to view detection history',
      });
    }

    const { page = 1, limit = 20, mediaType } = req.query;
    const filter = { userId: req.user._id };
    if (mediaType && ['image', 'video'].includes(mediaType)) {
      filter.mediaType = mediaType;
    }

    const skip = (Number(page) - 1) * Number(limit);
    const [detections, total] = await Promise.all([
      Detection.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(Number(limit))
        .lean(),
      Detection.countDocuments(filter),
    ]);

    res.json({
      success: true,
      page: Number(page),
      totalPages: Math.ceil(total / Number(limit)),
      total,
      detections,
    });
  } catch (err) {
    next(err);
  }
};

export const getDetectionById = async (req, res, next) => {
  try {
    const detection = await Detection.findById(req.params.id);
    if (!detection) {
      return res.status(404).json({
        success: false,
        message: 'Detection record not found',
      });
    }

    // Privacy check: guest detections can be viewed if known, user detections only by creator or admin
    if (
      detection.userId &&
      (!req.user || (req.user._id.toString() !== detection.userId.toString() && req.user.role !== 'admin'))
    ) {
      return res.status(403).json({
        success: false,
        message: 'Unauthorized access to this detection record',
      });
    }

    res.json({
      success: true,
      detection,
    });
  } catch (err) {
    next(err);
  }
};
