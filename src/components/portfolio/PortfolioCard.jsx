import React from 'react';
import { motion } from 'framer-motion';
import { FileText, ChevronRight, Code } from 'lucide-react';

export default function PortfolioCard({ item, onClick, index }) {
  const Icon = item.type === 'problem' ? Code : FileText;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      viewport={{ once: true }}
      whileHover={{ y: -8, transition: { duration: 0.2 } }}
      onClick={onClick}
      className="bg-white dark:bg-slate-800 rounded-2xl px-20 py-8 shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer border border-slate-100 dark:border-slate-700 group"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="p-3 bg-blue-50 dark:bg-blue-900 rounded-xl group-hover:bg-blue-100 dark:group-hover:bg-blue-800 transition-colors">
          <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        </div>
        <ChevronRight className="w-5 h-5 text-slate-400 dark:text-slate-500 group-hover:text-blue-600 dark:group-hover:text-blue-400 group-hover:translate-x-1 transition-all" />
      </div>

      <h3 className="text-3xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors leading-tight">
        {item.title}
      </h3>

      <p className="text-lg text-slate-600 dark:text-slate-300 leading-relaxed">
        {item.description}
      </p>

      <div className="mt-4 flex items-center text-base font-medium text-blue-600 dark:text-blue-400 group-hover:gap-2 transition-all">
        <span>Read More</span>
        <ChevronRight className="w-5 h-5 opacity-0 group-hover:opacity-100 transition-all" />
      </div>
    </motion.div>
  );
}