import React from 'react';
import { motion } from 'framer-motion';
import { FileText, ChevronRight } from 'lucide-react';

export default function PortfolioCard({ item, onClick, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      viewport={{ once: true }}
      whileHover={{ y: -8, transition: { duration: 0.2 } }}
      onClick={onClick}
      className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer border border-slate-100 group"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="p-3 bg-blue-50 rounded-xl group-hover:bg-blue-100 transition-colors">
          <FileText className="w-6 h-6 text-blue-600" />
        </div>
        <ChevronRight className="w-5 h-5 text-slate-400 group-hover:text-blue-600 group-hover:translate-x-1 transition-all" />
      </div>

      <h3 className="text-xl font-bold text-slate-900 mb-3 group-hover:text-blue-600 transition-colors">
        {item.title}
      </h3>
      
      <p className="text-slate-600 leading-relaxed line-clamp-3">
        {item.description}
      </p>

      <div className="mt-6 flex items-center text-sm font-medium text-blue-600 group-hover:gap-2 transition-all">
        <span>Read More</span>
        <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-all" />
      </div>
    </motion.div>
  );
}