import React from 'react';
import { motion } from 'framer-motion';
import { GraduationCap, Award, BookOpen, Target } from 'lucide-react';
import { easings, hoverLift, iconRotate } from '@/utils/motionConfig';

export default function AboutSection() {
  const highlights = [
    {
      icon: GraduationCap,
      title: 'Education',
      description: 'University of Oklahoma',
      detail: 'Major in Engineering Physics • GPA: 3.4'
    },
    {
      icon: Award,
      title: 'Achievements',
      description: 'Dean\'s List recipient',
      detail: 'Recognized for academic excellence'
    },
    {
      icon: BookOpen,
      title: 'Publications',
      description: '2024 ASC Conference Paper',
      detail: 'Published on structural health monitoring'
    },
    {
      icon: Target,
      title: 'Goals',
      description: 'Pursuing CS PhD opportunities',
      detail: 'Focus on Deep Learning & Autonomous Systems'
    }
  ];

  return (
    <section id="about" className="py-24 bg-white dark:bg-slate-900">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: easings.emphasis }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            About Me
          </h2>
          <motion.div
            initial={{ width: 0 }}
            whileInView={{ width: 80 }}
            transition={{ duration: 0.8, delay: 0.2, ease: easings.emphasis }}
            viewport={{ once: true }}
            className="h-1 bg-gradient-to-r from-blue-600 to-blue-500 dark:from-blue-500 dark:to-blue-400 mx-auto rounded-full"
          ></motion.div>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {highlights.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1, ease: easings.emphasis }}
              viewport={{ once: true, margin: "-50px" }}
              whileHover="hover"
              variants={hoverLift}
              className="bg-slate-50/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-2xl p-6 hover:bg-blue-50/80 dark:hover:bg-blue-950/40 transition-all duration-400 border border-slate-200/50 dark:border-slate-700/50 hover:border-blue-200 dark:hover:border-blue-800/50 shadow-md hover:shadow-float dark:shadow-glass dark:hover:shadow-glass-dark relative overflow-hidden group"
            >
              {/* Glass reflection overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-50/20 via-transparent to-white/10 dark:from-blue-900/10 dark:via-transparent dark:to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

              <div className="relative z-10">
                <motion.div
                  variants={iconRotate}
                  className="w-12 h-12 bg-blue-100/80 dark:bg-blue-950/50 backdrop-blur-sm rounded-xl flex items-center justify-center mb-4 group-hover:bg-blue-600 dark:group-hover:bg-blue-500 transition-all duration-300 shadow-sm"
                >
                  <item.icon className="w-6 h-6 text-blue-600 dark:text-blue-400 group-hover:text-white transition-colors duration-300" />
                </motion.div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">{item.title}</h3>
                <p className="text-slate-600 dark:text-slate-300 text-sm mb-1">{item.description}</p>
                <p className="text-slate-500 dark:text-slate-400 text-xs">{item.detail}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4, ease: easings.emphasis }}
          viewport={{ once: true }}
          whileHover={{ scale: 1.005 }}
          className="mt-16 bg-gradient-to-br from-blue-50/80 to-slate-50/80 dark:from-slate-800/80 dark:to-slate-700/80 backdrop-blur-glass rounded-3xl p-8 md:p-12 border border-slate-200/50 dark:border-slate-700/50 shadow-float dark:shadow-glass relative overflow-hidden group"
        >
          {/* Subtle animated gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-100/20 via-transparent to-slate-100/20 dark:from-blue-900/10 dark:via-transparent dark:to-slate-800/10 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none"></div>

          <div className="relative z-10">
            <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">Background</h3>
            <div className="space-y-4 text-slate-600 dark:text-slate-300 leading-relaxed">
              <motion.p
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.5 }}
                viewport={{ once: true }}
              >
                Throughout my undergraduate journey at the University of Oklahoma, I've cultivated a deep passion for deep learning and autonomous systems.
                My coursework in Engineering Physics has provided me with a strong theoretical foundation, while research
                experiences have honed my practical skills in applied machine learning, sensor design, and algorithm development.
              </motion.p>
              <motion.p
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.6 }}
                viewport={{ once: true }}
              >
                As an Undergraduate Researcher working under the guidance of Dr. Mrinal Saha and Postdoctoral Mentor Dr. Kuntal Maity,
                I've conducted experimental research on smart materials and developed deep learning algorithms for autonomous sensing technologies.
                My work on flexoelectric sensors and deep learning-based damage detection has contributed to cutting-edge research in structural health monitoring,
                resulting in a publication at the 39th Annual Technical Conference of the American Society for Composites.
              </motion.p>
              <motion.p
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.7 }}
                viewport={{ once: true }}
              >
                Beyond research, I founded Dirac Technologies where I led the development of autonomous robotic arms using imitation learning frameworks,
                and have been recognized on the Dean's List for academic excellence. As I prepare to graduate in December 2025, I'm eager to continue this journey at the doctoral level,
                where I can delve deeper into deep learning, reinforcement learning, and multimodal AI and collaborate with leading researchers in the field.
              </motion.p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}