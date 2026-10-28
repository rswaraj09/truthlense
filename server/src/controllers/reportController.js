import crypto from 'crypto';
import { Report } from '../models/Report.js';
import { Detection } from '../models/Detection.js';

const generateReferenceNumber = () => {
  const year = new Date().getFullYear();
  const random = crypto.randomBytes(4).toString('hex').toUpperCase();
  return `TL-${year}-${random}`;
};

export const createReport = async (req, res, next) => {
  try {
    const {
      detectionId,
      category,
      sourceUrl,
      description,
      evidenceReferences,
      contactName,
      contactEmail,
    } = req.body;

    if (!category || !description) {
      return res.status(400).json({
        success: false,
        message: 'Category and description are required for submitting a report',
      });
    }

    const referenceNumber = generateReferenceNumber();

    const submittedBy = {
      userId: req.user ? req.user._id : null,
      name: req.user ? req.user.name : contactName || 'Anonymous Submitter',
      email: req.user ? req.user.email : contactEmail || '',
    };

    const evidenceList = Array.isArray(evidenceReferences)
      ? evidenceReferences
      : evidenceReferences
      ? [evidenceReferences]
      : [];

    const report = await Report.create({
      referenceNumber,
      submittedBy,
      detectionId: detectionId || null,
      category,
      sourceUrl: sourceUrl || '',
      description,
      evidenceReferences: evidenceList,
      status: 'Submitted',
      auditLogs: [
        {
          action: 'REPORT_SUBMITTED',
          performedBy: submittedBy.name,
          notes: 'Report initially created and queued for forensics review',
        },
      ],
    });

    if (detectionId) {
      await Detection.findByIdAndUpdate(detectionId, { reportId: report._id });
    }

    res.status(201).json({
      success: true,
      referenceNumber: report.referenceNumber,
      reportId: report._id,
      status: report.status,
      message: 'Report submitted successfully. Keep your reference number for tracking.',
    });
  } catch (err) {
    next(err);
  }
};

export const getReportByRef = async (req, res, next) => {
  try {
    const { ref } = req.params;
    const report = await Report.findOne({ referenceNumber: ref.toUpperCase() })
      .populate('detectionId', 'mediaType prediction modelScore label createdAt')
      .lean();

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report with this reference number was not found',
      });
    }

    // Security check: private report details restricted unless submitter or admin
    const isOwner =
      req.user && report.submittedBy.userId && req.user._id.toString() === report.submittedBy.userId.toString();
    const isAdmin = req.user && req.user.role === 'admin';

    // Public view only sees sanitized status progress
    if (!isOwner && !isAdmin) {
      return res.json({
        success: true,
        report: {
          referenceNumber: report.referenceNumber,
          category: report.category,
          status: report.status,
          removalRequestStatus: report.removalRequestStatus,
          createdAt: report.createdAt,
          updatedAt: report.updatedAt,
        },
      });
    }

    res.json({
      success: true,
      report,
    });
  } catch (err) {
    next(err);
  }
};

export const getUserReports = async (req, res, next) => {
  try {
    const reports = await Report.find({ 'submittedBy.userId': req.user._id })
      .populate('detectionId', 'mediaType prediction modelScore')
      .sort({ createdAt: -1 })
      .lean();

    res.json({
      success: true,
      reports,
    });
  } catch (err) {
    next(err);
  }
};

// Admin endpoints
export const getAllReports = async (req, res, next) => {
  try {
    const { status, category, search, page = 1, limit = 20 } = req.query;
    const filter = {};

    if (status) filter.status = status;
    if (category) filter.category = category;
    if (search) {
      filter.$or = [
        { referenceNumber: { $regex: search, $options: 'i' } },
        { description: { $regex: search, $options: 'i' } },
        { sourceUrl: { $regex: search, $options: 'i' } },
      ];
    }

    const skip = (Number(page) - 1) * Number(limit);
    const [reports, total] = await Promise.all([
      Report.find(filter)
        .populate('detectionId', 'mediaType prediction modelScore label createdAt')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(Number(limit))
        .lean(),
      Report.countDocuments(filter),
    ]);

    res.json({
      success: true,
      page: Number(page),
      totalPages: Math.ceil(total / Number(limit)),
      total,
      reports,
    });
  } catch (err) {
    next(err);
  }
};

export const updateReportStatus = async (req, res, next) => {
  try {
    const { id } = req.params;
    const { status, reviewerNotes } = req.body;

    const validStatuses = [
      'Submitted',
      'Under review',
      'Additional information required',
      'Removal requested',
      'Resolved',
      'Rejected',
    ];

    if (!validStatuses.includes(status)) {
      return res.status(400).json({
        success: false,
        message: `Invalid status. Must be one of: ${validStatuses.join(', ')}`,
      });
    }

    const report = await Report.findById(id);
    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found',
      });
    }

    const prevStatus = report.status;
    report.status = status;
    if (reviewerNotes !== undefined) {
      report.reviewerNotes = reviewerNotes;
    }
    report.reviewedBy = req.user._id;

    report.auditLogs.push({
      action: `STATUS_CHANGED_${status.toUpperCase().replace(/\s+/g, '_')}`,
      performedBy: `${req.user.name} (Admin)`,
      notes: `Status changed from "${prevStatus}" to "${status}". Notes: ${reviewerNotes || 'None'}`,
    });

    await report.save();

    res.json({
      success: true,
      message: 'Report status updated successfully',
      report,
    });
  } catch (err) {
    next(err);
  }
};

export const updateRemovalRequest = async (req, res, next) => {
  try {
    const { id } = req.params;
    const { removalRequestStatus, platformTarget, notes } = req.body;

    const validRemovalStatuses = [
      'Not requested',
      'Draft prepared',
      'Sent to platform',
      'Platform acknowledged',
      'Content removed',
      'Request declined',
    ];

    if (!validRemovalStatuses.includes(removalRequestStatus)) {
      return res.status(400).json({
        success: false,
        message: `Invalid removal request status. Must be one of: ${validRemovalStatuses.join(', ')}`,
      });
    }

    const report = await Report.findById(id);
    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found',
      });
    }

    const prevStatus = report.removalRequestStatus;
    report.removalRequestStatus = removalRequestStatus;
    if (platformTarget) report.platformTarget = platformTarget;

    report.auditLogs.push({
      action: `REMOVAL_STATUS_${removalRequestStatus.toUpperCase().replace(/\s+/g, '_')}`,
      performedBy: `${req.user.name} (Admin)`,
      notes: `Removal request updated from "${prevStatus}" to "${removalRequestStatus}". Target platform: ${
        platformTarget || report.platformTarget || 'N/A'
      }. Notes: ${notes || 'None'}`,
    });

    // If removal was completed or requested, sync main status if appropriate
    if (removalRequestStatus === 'Content removed') {
      report.status = 'Resolved';
    } else if (removalRequestStatus === 'Sent to platform') {
      report.status = 'Removal requested';
    }

    await report.save();

    res.json({
      success: true,
      message: 'Removal request status updated successfully',
      report,
    });
  } catch (err) {
    next(err);
  }
};
