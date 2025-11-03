import React from 'react';
import { motion } from 'framer-motion';
import { GraduationCap, Award, BookOpen, Target } from 'lucide-react';

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
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            About Me
          </h2>
          <div className="w-20 h-1 bg-blue-600 dark:bg-blue-500 mx-auto rounded-full"></div>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {highlights.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-slate-50 dark:bg-slate-800 rounded-2xl p-6 hover:bg-blue-50 dark:hover:bg-blue-950/30 transition-all duration-300 group"
            >
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-950 rounded-xl flex items-center justify-center mb-4 group-hover:bg-blue-600 transition-colors">
                <item.icon className="w-6 h-6 text-blue-600 dark:text-blue-400 group-hover:text-white transition-colors" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">{item.title}</h3>
              <p className="text-slate-600 dark:text-slate-300 text-sm mb-1">{item.description}</p>
              <p className="text-slate-500 dark:text-slate-400 text-xs">{item.detail}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-16 bg-gradient-to-br from-blue-50 to-slate-50 dark:from-slate-800 dark:to-slate-700 rounded-3xl p-8 md:p-12"
        >
          <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">Background</h3>
          <div className="space-y-4 text-slate-600 dark:text-slate-300 leading-relaxed">
            <p>
              Throughout my undergraduate journey at the University of Oklahoma, I've cultivated a deep passion for deep learning and autonomous systems.
              My coursework in Engineering Physics has provided me with a strong theoretical foundation, while research
              experiences have honed my practical skills in applied machine learning, sensor design, and algorithm development.
            </p>
            <p>
              As an Undergraduate Researcher working under the guidance of Dr. Mrinal Saha and Postdoctoral Mentor Dr. Kuntal Maity,
              I've conducted experimental research on smart materials and developed deep learning algorithms for autonomous sensing technologies.
              My work on flexoelectric sensors and deep learning-based damage detection has contributed to cutting-edge research in structural health monitoring,
              resulting in a publication at the 39th Annual Technical Conference of the American Society for Composites.
            </p>
            <p>
              Beyond research, I founded Dirac Technologies where I led the development of autonomous robotic arms using imitation learning frameworks,
              and have been recognized on the Dean's List for academic excellence. As I prepare to graduate in December 2025, I'm eager to continue this journey at the doctoral level,
              where I can delve deeper into deep learning, reinforcement learning, and multimodal AI and collaborate with leading researchers in the field.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}