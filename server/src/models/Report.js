import mongoose from 'mongoose';

const reportSchema = new mongoose.Schema(
  {
    referenceNumber: {
      type: String,
      required: true,
      unique: true,
      index: true,
      uppercase: true,
    },
    submittedBy: {
      userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        default: null,
      },
      name: {
        type: String,
        default: 'Anonymous User',
      },
      email: {
        type: String,
        default: '',
      },
    },
    detectionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Detection',
      default: null,
    },
    category: {
      type: String,
      required: true,
      enum: [
        'Suspected AI impersonation',
        'Non-consensual manipulated imagery',
        'Suspected defamatory content',
        'Misleading viral media',
        'Other harmful manipulated media',
      ],
    },
    sourceUrl: {
      type: String,
      trim: true,
      default: '',
    },
    description: {
      type: String,
      required: [true, 'Report description is required'],
      maxlength: [3000, 'Description cannot exceed 3000 characters'],
    },
    evidenceReferences: [
      {
        type: String,
        trim: true,
      },
    ],
    status: {
      type: String,
      enum: [
        'Submitted',
        'Under review',
        'Additional information required',
        'Removal requested',
        'Resolved',
        'Rejected',
      ],
      default: 'Submitted',
      index: true,
    },
    reviewerNotes: {
      type: String,
      default: '',
    },
    reviewedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      default: null,
    },
    removalRequestStatus: {
      type: String,
      enum: [
        'Not requested',
        'Draft prepared',
        'Sent to platform',
        'Platform acknowledged',
        'Content removed',
        'Request declined',
      ],
      default: 'Not requested',
    },
    platformTarget: {
      type: String,
      default: '',
    },
    auditLogs: [
      {
        action: {
          type: String,
          required: true,
        },
        performedBy: {
          type: String,
          required: true,
        },
        performedAt: {
          type: Date,
          default: Date.now,
        },
        notes: String,
      },
    ],
  },
  {
    timestamps: true,
  }
);

reportSchema.index({ createdAt: -1 });

export const Report = mongoose.model('Report', reportSchema);
