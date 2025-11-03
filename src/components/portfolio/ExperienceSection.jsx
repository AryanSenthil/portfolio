import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { base44 } from '@/api/base44Client';
import { Briefcase } from 'lucide-react';

export default function ExperienceSection() {
  const { data: experiences = [] } = useQuery({
    queryKey: ['experiences'],
    queryFn: async () => {
      const allExperiences = await base44.entities.Experience.list('order');
      // Filter out involvement positions (Community service chair, Executive Board Member)
      return allExperiences.filter(exp =>
        !exp.position?.toLowerCase().includes('community service chair') &&
        !exp.position?.toLowerCase().includes('executive board member')
      );
    },
    initialData: []
  });

  return (
    <section id="experience" className="py-24 bg-slate-900 dark:bg-slate-950 text-white">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4 dark:text-white">
            Experience
          </h2>
          <div className="w-20 h-1 bg-blue-500 dark:bg-blue-400 rounded-full"></div>
        </motion.div>

        <div className="relative">
          {/* Timeline line - connects between circle centers only */}
          {experiences.length > 1 && (
            <div
              className="absolute left-6 w-0.5 bg-slate-700 dark:bg-slate-600 hidden md:block"
              style={{
                top: 'calc(3rem + 1.5rem)',
                bottom: `calc(3rem + 1.5rem)`
              }}
            ></div>
          )}

          <div className="space-y-12">
            {experiences.map((exp, index) => (
              <motion.div
                key={exp.id}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="relative"
              >
                {/* Timeline dot - centered with the content box */}
                <div className="absolute left-0 top-6 w-12 h-12 bg-blue-500 dark:bg-blue-400 rounded-full items-center justify-center hidden md:flex">
                  <div className="w-3 h-3 bg-white dark:bg-slate-900 rounded-full"></div>
                </div>

                {/* Content */}
                <div className="md:ml-20 bg-slate-800 dark:bg-slate-900 rounded-2xl p-8 hover:bg-slate-750 dark:hover:bg-slate-850 transition-all border border-slate-700 dark:border-slate-600">
                  <div className="flex items-start gap-4 mb-4">
                    {exp.company_logo ? (
                      <img
                        src={exp.company_logo}
                        alt={exp.company}
                        className="w-12 h-12 rounded-lg object-contain bg-white dark:bg-slate-800 p-2"
                      />
                    ) : (
                      <div className="w-12 h-12 bg-slate-700 dark:bg-slate-600 rounded-lg flex items-center justify-center">
                        <Briefcase className="w-6 h-6 text-slate-400 dark:text-slate-300" />
                      </div>
                    )}

                    <div className="flex-1">
                      <h3 className="text-2xl font-bold text-white dark:text-white mb-1">
                        {exp.position}
                      </h3>
                      <p className="text-lg text-blue-400 dark:text-blue-300 font-medium mb-2">
                        {exp.company}
                      </p>
                      <p className="text-sm text-slate-400 dark:text-slate-300">
                        {exp.start_date} – {exp.end_date} {exp.location && `· ${exp.location}`}
                      </p>
                    </div>
                  </div>

                  {exp.description && exp.description.length > 0 && (
                    <ul className="space-y-2 mt-4">
                      {exp.description.map((item, idx) => (
                        <li key={idx} className="text-slate-300 dark:text-slate-200 flex items-start gap-3">
                          <span className="text-blue-400 dark:text-blue-300 mt-1.5">•</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </motion.div>
            ))}
          </div>

          {experiences.length === 0 && (
            <div className="text-center py-12">
              <p className="text-slate-400 dark:text-slate-300 mb-4">No experience items yet.</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Add experience items through the dashboard.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}