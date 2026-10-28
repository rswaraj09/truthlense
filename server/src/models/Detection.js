import mongoose from 'mongoose';

const detectionSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      default: null,
      index: true,
    },
    mediaType: {
      type: String,
      enum: ['image', 'video'],
      required: true,
    },
    originalFilename: {
      type: String,
      required: true,
      trim: true,
    },
    fileSize: {
      type: Number,
      required: true,
    },
    mimeType: {
      type: String,
      required: true,
    },
    prediction: {
      type: String,
      enum: ['ai_generated', 'authentic', 'manipulated', 'inconclusive'],
      required: true,
    },
    label: {
      type: String,
      required: true,
    },
    modelScore: {
      type: Number,
      required: true,
      min: 0,
      max: 1,
    },
    confidenceLevel: {
      type: String,
      enum: ['high', 'medium', 'low', 'inconclusive'],
      default: 'medium',
    },
    processingTimeMs: {
      type: Number,
      required: true,
    },
    isPreliminaryScreening: {
      type: Boolean,
      default: false,
    },
    forensicMetrics: {
      type: mongoose.Schema.Types.Mixed,
      default: {},
    },
    framesAnalysis: [
      {
        frameIndex: Number,
        timestampSec: Number,
        timestampFormatted: String,
        score: Number,
        prediction: String,
        flagged: Boolean,
      },
    ],
    limitationDisclaimer: {
      type: String,
      default: '',
    },
    reportId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Report',
      default: null,
    },
  },
  {
    timestamps: true,
  }
);

detectionSchema.index({ createdAt: -1 });

export const Detection = mongoose.model('Detection', detectionSchema);
