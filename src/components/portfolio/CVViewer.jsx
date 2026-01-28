import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function CVViewer({ onClose }) {
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = '/portfolio/aryansenthilresume.pdf';
    link.download = 'Aryan_Senthil_Resume.pdf';
    link.click();
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", duration: 0.5 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-white dark:bg-slate-900 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
          style={{
            width: '210mm',
            maxWidth: '90vw',
            height: '95vh',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 shrink-0">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">Resume</h2>
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                className="text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                <Download className="w-4 h-4 mr-2" />
                Download PDF
              </Button>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="rounded-full hover:bg-slate-200 dark:hover:bg-slate-700"
            >
              <X className="w-5 h-5 text-slate-700 dark:text-white" />
            </Button>
          </div>

          {/* Resume Content - HTML Viewer */}
          <div className="flex-1 overflow-hidden bg-slate-100 dark:bg-slate-950">
            <iframe
              src="/portfolio/cv/cv.html"
              title="Resume"
              className="w-full h-full border-0"
              style={{
                backgroundColor: '#f5f5f5',
              }}
            />
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
