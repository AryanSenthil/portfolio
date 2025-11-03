import React from 'react';
import { motion } from 'framer-motion';
import { FileText, ChevronRight, Code } from 'lucide-react';
import { easings, hoverLift, iconRotate } from '@/utils/motionConfig';

export default function PortfolioCard({ item, onClick, index }) {
  const Icon = item.type === 'problem' ? Code : FileText;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.1, ease: easings.emphasis }}
      viewport={{ once: true, margin: "-50px" }}
      whileHover="hover"
      whileTap={{ scale: 0.98 }}
      variants={hoverLift}
      onClick={onClick}
      className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-2xl px-20 py-8 shadow-float hover:shadow-float-dark dark:shadow-glass dark:hover:shadow-glass-dark transition-all duration-400 cursor-pointer border border-slate-100/50 dark:border-slate-700/50 hover:border-blue-200 dark:hover:border-blue-800/50 group relative overflow-hidden"
    >
      {/* Glass reflection overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/20 via-transparent to-white/10 dark:from-blue-900/10 dark:via-transparent dark:to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-3">
          <motion.div
            variants={iconRotate}
            className="p-3 bg-blue-50/80 dark:bg-blue-900/50 backdrop-blur-sm rounded-xl group-hover:bg-blue-100 dark:group-hover:bg-blue-800/70 transition-all duration-300 shadow-sm"
          >
            <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          </motion.div>
          <motion.div
            animate={{ x: [0, 4, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          >
            <ChevronRight className="w-5 h-5 text-slate-400 dark:text-slate-500 group-hover:text-blue-600 dark:group-hover:text-blue-400 group-hover:translate-x-1 transition-all duration-300" />
          </motion.div>
        </div>

        <h3 className="text-3xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300 leading-tight">
          {item.title}
        </h3>

        <p className="text-lg text-slate-600 dark:text-slate-300 leading-relaxed">
          {item.description}
        </p>

        <motion.div
          className="mt-4 flex items-center text-base font-medium text-blue-600 dark:text-blue-400 transition-all duration-300"
          initial={{ gap: 0 }}
          whileHover={{ gap: 8 }}
        >
          <span>Read More</span>
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            whileHover={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <ChevronRight className="w-5 h-5" />
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  );
}