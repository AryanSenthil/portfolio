import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function HTMLViewer({ item, onClose }) {
  useEffect(() => {
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  if (!item) return null;

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
          className="bg-white rounded-3xl shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-200 bg-slate-50">
            <div>
              <h2 className="text-2xl font-bold text-slate-900">{item.title}</h2>
              <p className="text-sm text-slate-500 mt-1">{item.description}</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="rounded-full hover:bg-slate-200"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-8">
            {item.content_html ? (
              <div 
                className="prose prose-lg prose-slate max-w-none prose-headings:text-slate-900 prose-p:text-slate-600 prose-a:text-blue-600 prose-strong:text-slate-900 prose-ul:text-slate-600 prose-ol:text-slate-600"
                dangerouslySetInnerHTML={{ __html: item.content_html }}
              />
            ) : item.html_file_url ? (
              <iframe
                src={item.html_file_url}
                className="w-full h-full min-h-[600px] border-0"
                title={item.title}
              />
            ) : (
              <div className="text-center py-12">
                <p className="text-slate-500">No content available for this item.</p>
                <p className="text-sm text-slate-400 mt-2">
                  Add HTML content or upload an HTML file through the dashboard.
                </p>
              </div>
            )}
          </div>

          {/* Footer hint */}
          <div className="p-4 bg-slate-50 text-center border-t border-slate-200">
            <p className="text-xs text-slate-500">
              Press ESC or click outside to close
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}