import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { base44 } from '@/api/base44Client';
import { Briefcase } from 'lucide-react';
import { easings, hoverLift } from '@/utils/motionConfig';

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
    <section id="experience" className="py-16 sm:py-24 bg-slate-900 dark:bg-slate-950 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: easings.emphasis }}
          viewport={{ once: true }}
          className="mb-16 text-center"
        >
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 dark:text-white">
            Experience
          </h2>
          <motion.div
            initial={{ width: 0 }}
            whileInView={{ width: 80 }}
            transition={{ duration: 0.8, delay: 0.2, ease: easings.emphasis }}
            viewport={{ once: true }}
            className="h-1 bg-gradient-to-r from-blue-500 to-blue-400 rounded-full mx-auto mb-6"
          ></motion.div>
          <p className="text-lg text-slate-400 dark:text-slate-300 max-w-2xl mx-auto">
            Professional journey spanning research, development, and leadership
          </p>
        </motion.div>

        <div className="relative">
          {/* Timeline line - animated drawing effect */}
          {experiences.length > 1 && (
            <motion.div
              initial={{ scaleY: 0, opacity: 0 }}
              whileInView={{ scaleY: 1, opacity: 1 }}
              transition={{ duration: 1.2, delay: 0.3, ease: easings.emphasis }}
              viewport={{ once: true }}
              className="absolute left-6 w-0.5 bg-gradient-to-b from-blue-500/50 via-slate-700 to-slate-700 dark:from-blue-400/50 dark:via-slate-600 dark:to-slate-600 hidden md:block origin-top"
              style={{
                top: 'calc(3rem + 1.5rem)',
                height: `calc(100% - ${experiences.length * 3}rem - 9rem)`
              }}
            ></motion.div>
          )}

          <div className="space-y-12">
            {experiences.map((exp, index) => (
              <motion.div
                key={exp.id}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1, ease: easings.emphasis }}
                viewport={{ once: true, margin: "-50px" }}
                className="relative"
              >
                {/* Timeline dot - pulse animation on scroll */}
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  whileInView={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.1 + 0.3, type: "spring", stiffness: 200 }}
                  viewport={{ once: true }}
                  className="absolute left-0 top-6 w-12 h-12 bg-blue-500 dark:bg-blue-400 rounded-full items-center justify-center hidden md:flex shadow-glow-blue-lg"
                >
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                    className="w-3 h-3 bg-white dark:bg-slate-900 rounded-full"
                  ></motion.div>
                </motion.div>

                {/* Content with glass morphism */}
                <motion.div
                  initial="rest"
                  whileHover="hover"
                  variants={hoverLift}
                  className="md:ml-20 bg-slate-800/80 dark:bg-slate-900/80 backdrop-blur-glass rounded-2xl p-8 transition-all duration-400 border border-slate-700/50 dark:border-slate-600/50 hover:border-slate-600 dark:hover:border-slate-500 relative overflow-hidden group"
                >
                  {/* Glass reflection overlay */}
                  <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

                  <div className="relative z-10">
                    <div className="flex items-start gap-4 mb-4">
                      {exp.company_logo ? (
                        <motion.img
                          whileHover={{ scale: 1.1, rotate: 3 }}
                          transition={{ duration: 0.3 }}
                          src={exp.company_logo}
                          alt={exp.company}
                          className="w-12 h-12 rounded-lg object-contain bg-white/90 dark:bg-slate-800/90 p-2 backdrop-blur-sm"
                        />
                      ) : (
                        <motion.div
                          whileHover={{ scale: 1.1, rotate: 3 }}
                          transition={{ duration: 0.3 }}
                          className="w-12 h-12 bg-slate-700/50 dark:bg-slate-600/50 backdrop-blur-sm rounded-lg flex items-center justify-center"
                        >
                          <Briefcase className="w-6 h-6 text-slate-400 dark:text-slate-300" />
                        </motion.div>
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
                          <motion.li
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: idx * 0.05 }}
                            viewport={{ once: true }}
                            className="text-slate-300 dark:text-slate-200 flex items-start gap-3"
                          >
                            <span className="text-blue-400 dark:text-blue-300 mt-1.5">•</span>
                            <span>{item}</span>
                          </motion.li>
                        ))}
                      </ul>
                    )}
                  </div>
                </motion.div>
              </motion.div>
            ))}
          </div>

          {experiences.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-center py-12"
            >
              <p className="text-slate-400 dark:text-slate-300 mb-4">No experience items yet.</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Add experience items through the dashboard.
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </section>
  );
}