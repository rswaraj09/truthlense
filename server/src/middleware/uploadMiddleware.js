import multer from 'multer';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';

const UPLOAD_DIR = path.resolve('uploads');
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Storage engine with random unique filenames
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const randomHex = crypto.randomBytes(16).toString('hex');
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${Date.now()}-${randomHex}${ext}`);
  },
});

// Magic bytes validation for safe uploads
export const validateFileSignature = (filePath, expectedType) => {
  try {
    const buffer = Buffer.alloc(12);
    const fd = fs.openSync(filePath, 'r');
    fs.readSync(fd, buffer, 0, 12, 0);
    fs.closeSync(fd);

    const hex = buffer.toString('hex').toLowerCase();

    if (expectedType === 'image') {
      // JPEG: ffd8ff
      if (hex.startsWith('ffd8ff')) return true;
      // PNG: 89504e470d0a1a0a
      if (hex.startsWith('89504e470d0a1a0a')) return true;
      // WebP: RIFF....WEBP (52494646....57454250)
      if (hex.startsWith('52494646') && hex.slice(16, 24) === '57454250') return true;
      // GIF: 47494638
      if (hex.startsWith('47494638')) return true;
      return false;
    }

    if (expectedType === 'video') {
      // MP4 / MOV / M4V (ftyp box)
      if (hex.slice(8, 16) === '66747970') return true;
      // WebM / MKV (1a45dfa3)
      if (hex.startsWith('1a45dfa3')) return true;
      // AVI: RIFF....AVI (52494646....41564920)
      if (hex.startsWith('52494646') && hex.slice(16, 24) === '41564920') return true;
      // QuickTime moov/mdat
      if (hex.slice(8, 16) === '6d6f6f76' || hex.slice(8, 16) === '6d646174') return true;
      return false;
    }

    return false;
  } catch (err) {
    return false;
  }
};

// Safe file cleanup helper
export const cleanupFile = (filePath) => {
  if (!filePath) return;
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (err) {
    console.error(`[Upload] Failed to clean up temporary file ${filePath}:`, err.message);
  }
};

// Image upload config
export const uploadImage = multer({
  storage,
  limits: {
    fileSize: 25 * 1024 * 1024, // 25 MB
    files: 1,
  },
  fileFilter: (req, file, cb) => {
    const allowedExts = /jpeg|jpg|png|webp/;
    const extname = allowedExts.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedExts.test(file.mimetype);

    if (extname && mimetype) {
      return cb(null, true);
    }
    cb(new Error('Invalid image format. Allowed: JPG, JPEG, PNG, WEBP'));
  },
});

// Video upload config
export const uploadVideo = multer({
  storage,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100 MB
    files: 1,
  },
  fileFilter: (req, file, cb) => {
    const allowedExts = /mp4|webm|avi|mov/;
    const extname = allowedExts.test(path.extname(file.originalname).toLowerCase());
    const mimetype = /video\/(mp4|webm|x-msvideo|quicktime)/.test(file.mimetype);

    if (extname || mimetype) {
      return cb(null, true);
    }
    cb(new Error('Invalid video format. Allowed: MP4, WebM, AVI, MOV'));
  },
});
