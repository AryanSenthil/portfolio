import React from 'react';
import { motion } from 'framer-motion';
import {
  Code2,
  Brain,
  Bot,
  Cpu,
  Database,
  Terminal,
  Box,
  Calculator,
  Cloud
} from 'lucide-react';
import { easings, hoverLift, iconRotate } from '@/utils/motionConfig';

export default function SkillsSection() {
  const skillCategories = [
    {
      icon: Code2,
      title: 'Programming Languages',
      skills: ['Python', 'C++', 'R', 'MATLAB', 'Mathematica']
    },
    {
      icon: Brain,
      title: 'Machine Learning Frameworks',
      skills: ['PyTorch', 'TensorFlow', 'Keras', 'scikit-learn']
    },
    {
      icon: Bot,
      title: 'AI Agents & LLM Tools',
      skills: ['LangChain', 'LangGraph', 'LlamaIndex', 'RAG Systems']
    },
    {
      icon: Cpu,
      title: 'Robotics & Embedded Systems',
      skills: ['ROS 2', 'NVIDIA Jetson', 'CUDA']
    },
    {
      icon: Database,
      title: 'Data Engineering & Databases',
      skills: ['SQL', 'Redis', 'Weaviate', 'PostgreSQL']
    },
    {
      icon: Terminal,
      title: 'DevOps & Development Tools',
      skills: ['Linux', 'Git', 'Docker', 'CI/CD']
    },
    {
      icon: Box,
      title: 'CAD & Design',
      skills: ['SolidWorks', 'Shapr3D']
    },
    {
      icon: Calculator,
      title: 'Scientific Computing & Documentation',
      skills: ['LaTeX', 'Jupyter', 'NumPy', 'Pandas']
    },
    {
      icon: Cloud,
      title: 'Cloud & Infrastructure',
      skills: ['AWS', 'GCP', 'Azure', 'Kubernetes']
    }
  ];

  return (
    <section id="skills" className="py-24 bg-white dark:bg-slate-900">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: easings.emphasis }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            Skills
          </h2>
          <motion.div
            initial={{ width: 0 }}
            whileInView={{ width: 80 }}
            transition={{ duration: 0.8, delay: 0.2, ease: easings.emphasis }}
            viewport={{ once: true }}
            className="h-1 bg-gradient-to-r from-blue-600 to-blue-500 dark:from-blue-500 dark:to-blue-400 mx-auto rounded-full mb-6"
          ></motion.div>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Technical expertise across machine learning, robotics, and software development
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {skillCategories.map((category, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1, ease: easings.emphasis }}
              viewport={{ once: true, margin: "-50px" }}
              whileHover="hover"
              variants={hoverLift}
              className="bg-slate-50/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-2xl p-6 shadow-float hover:shadow-float-dark dark:shadow-glass dark:hover:shadow-glass-dark transition-all duration-400 border-t-4 border-blue-600 dark:border-blue-500 border border-slate-200/50 dark:border-slate-700/50 hover:border-slate-300 dark:hover:border-slate-600 relative overflow-hidden group"
            >
              {/* Glass reflection overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-blue-50/10 dark:from-white/5 dark:to-blue-900/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

              <div className="relative z-10">
                <div className="flex items-start gap-4 mb-4">
                  <motion.div
                    variants={iconRotate}
                    className="p-3 bg-blue-50/80 dark:bg-blue-900/50 backdrop-blur-sm rounded-xl group-hover:bg-blue-100 dark:group-hover:bg-blue-800/70 transition-all duration-300 flex-shrink-0 shadow-sm"
                  >
                    <category.icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                  </motion.div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white leading-tight pt-2">
                    {category.title}
                  </h3>
                </div>

                <div className="flex flex-wrap gap-2">
                  {category.skills.map((skill, skillIndex) => (
                    <motion.span
                      key={skillIndex}
                      initial={{ opacity: 0, scale: 0.8 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      transition={{
                        duration: 0.3,
                        delay: index * 0.1 + skillIndex * 0.05,
                        ease: easings.emphasis
                      }}
                      viewport={{ once: true }}
                      whileHover={{
                        scale: 1.05,
                        y: -2,
                        transition: { duration: 0.2 }
                      }}
                      whileTap={{ scale: 0.95 }}
                      className="px-3 py-1.5 bg-white/90 dark:bg-slate-700/80 backdrop-blur-sm text-slate-700 dark:text-slate-200 rounded-full text-sm font-medium border border-slate-200 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-slate-600/90 hover:text-blue-700 dark:hover:text-blue-300 transition-all duration-300 cursor-default shadow-sm hover:shadow-md"
                    >
                      {skill}
                    </motion.span>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
