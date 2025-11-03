import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { base44 } from '@/api/base44Client';
import { Users, Heart } from 'lucide-react';

export default function InvolvementSection() {
  const { data: involvements = [] } = useQuery({
    queryKey: ['involvements'],
    queryFn: async () => {
      const allExperiences = await base44.entities.Experience.list('order');
      // Filter for Community service chair and Executive Board Member
      return allExperiences.filter(exp =>
        exp.position?.toLowerCase().includes('community service chair') ||
        exp.position?.toLowerCase().includes('executive board member')
      );
    },
    initialData: []
  });

  return (
    <section id="involvement" className="py-24 bg-white dark:bg-slate-900">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-slate-900 dark:text-white">
            Community Involvement
          </h2>
          <div className="w-20 h-1 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {involvements.map((involvement, index) => (
            <motion.div
              key={involvement.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-slate-50 dark:bg-slate-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all border-2 border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-400"
            >
              <div className="flex items-start gap-4 mb-6">
                {involvement.company_logo ? (
                  <img
                    src={involvement.company_logo}
                    alt={involvement.company}
                    className="w-16 h-16 rounded-xl object-contain bg-white dark:bg-slate-900 p-2 shadow-md"
                  />
                ) : (
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 dark:from-blue-400 dark:to-blue-500 rounded-xl flex items-center justify-center shadow-md">
                    <Users className="w-8 h-8 text-white" />
                  </div>
                )}

                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
                    {involvement.position}
                  </h3>
                  <p className="text-blue-600 dark:text-blue-400 font-semibold mb-1">
                    {involvement.company}
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-400">
                    {involvement.start_date} – {involvement.end_date} {involvement.location && `· ${involvement.location}`}
                  </p>
                </div>
              </div>

              {involvement.description && involvement.description.length > 0 && (
                <ul className="space-y-3">
                  {involvement.description.map((item, idx) => (
                    <li key={idx} className="text-slate-700 dark:text-slate-300 flex items-start gap-3">
                      <Heart className="w-5 h-5 text-blue-500 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              )}
            </motion.div>
          ))}
        </div>

        {involvements.length === 0 && (
          <div className="text-center py-12">
            <p className="text-slate-400 dark:text-slate-300 mb-4">No involvement activities yet.</p>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Add community and leadership activities through the dashboard.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
