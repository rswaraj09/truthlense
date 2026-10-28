import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://127.0.0.1:8000';

export class AIService {
  /**
   * Check health and model status of Python AI service
   */
  static async checkHealth() {
    try {
      const response = await axios.get(`${AI_SERVICE_URL}/health`, {
        timeout: 5000,
      });
      return response.data;
    } catch (err) {
      console.warn(`[AIService] Health check failed: ${err.message}`);
      return {
        success: false,
        status: 'unreachable',
        error: err.message,
      };
    }
  }

  /**
   * Send image to Python service for PyTorch inference
   */
  static async predictImage(filePath, originalFilename) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), {
      filename: originalFilename,
    });

    try {
      const response = await axios.post(`${AI_SERVICE_URL}/predict/image`, form, {
        headers: {
          ...form.getHeaders(),
        },
        maxContentLength: 50 * 1024 * 1024,
        maxBodyLength: 50 * 1024 * 1024,
        timeout: 60000,
      });

      return response.data;
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.response?.data?.message || err.message;
      throw new Error(`AI inference failed: ${errMsg}`);
    }
  }

  /**
   * Send video to Python service for frame-by-frame screening
   */
  static async predictVideo(filePath, originalFilename) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), {
      filename: originalFilename,
    });

    try {
      const response = await axios.post(`${AI_SERVICE_URL}/predict/video`, form, {
        headers: {
          ...form.getHeaders(),
        },
        maxContentLength: 150 * 1024 * 1024,
        maxBodyLength: 150 * 1024 * 1024,
        timeout: 180000, // 3 minutes for video processing
      });

      return response.data;
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.response?.data?.message || err.message;
      throw new Error(`AI video screening failed: ${errMsg}`);
    }
  }
}
