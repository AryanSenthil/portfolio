import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

export default function CVViewer({ onClose }) {
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = '/cv.pdf';
    link.download = 'Aryan_Yamini_Senthil_CV.pdf';
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
          className="bg-white rounded-3xl shadow-2xl overflow-hidden flex flex-col"
          style={{
            width: '210mm',
            maxWidth: '90vw',
            height: '95vh',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50 shrink-0">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-slate-900">Curriculum Vitae</h2>
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                className="text-slate-600 hover:bg-slate-100"
              >
                <Download className="w-4 h-4 mr-2" />
                Download PDF
              </Button>
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

          {/* CV Content - Zoomable PDF-like background */}
          <div className="flex-1 overflow-hidden bg-black relative">
            <TransformWrapper
              initialScale={1}
              minScale={0.5}
              maxScale={3}
              wheel={{ step: 0.1 }}
              pinch={{ step: 5 }}
              doubleClick={{ mode: 'reset' }}
            >
              {({ zoomIn, zoomOut, resetTransform }) => (
                <>
                  {/* Zoom Controls */}
                  <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => zoomIn()}
                      className="bg-white hover:bg-slate-100 shadow-lg"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => zoomOut()}
                      className="bg-white hover:bg-slate-100 shadow-lg"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => resetTransform()}
                      className="bg-white hover:bg-slate-100 shadow-lg"
                      title="Reset Zoom"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </Button>
                  </div>

                  {/* Transformable Content */}
                  <TransformComponent
                    wrapperStyle={{
                      width: '100%',
                      height: '100%',
                    }}
                    contentStyle={{
                      width: '100%',
                      height: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: '2rem',
                    }}
                  >
                    <div className="flex flex-col gap-4">
                      <img
                        src="/cv/images/cv-1.png"
                        alt="CV Page 1"
                        className="bg-white shadow-2xl"
                        style={{ width: '210mm', display: 'block' }}
                      />
                      <img
                        src="/cv/images/cv-2.png"
                        alt="CV Page 2"
                        className="bg-white shadow-2xl"
                        style={{ width: '210mm', display: 'block' }}
                      />
                    </div>
                  </TransformComponent>
                </>
              )}
            </TransformWrapper>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
