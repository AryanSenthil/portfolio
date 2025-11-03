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
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            Skills
          </h2>
          <div className="w-20 h-1 bg-blue-600 dark:bg-blue-500 mx-auto rounded-full mb-6"></div>
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
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ y: -8, transition: { duration: 0.2 } }}
              className="bg-slate-50 dark:bg-slate-800 rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all duration-300 border-t-4 border-blue-600 dark:border-blue-500 group"
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="p-3 bg-blue-50 dark:bg-blue-900 rounded-xl group-hover:bg-blue-100 dark:group-hover:bg-blue-800 transition-colors flex-shrink-0">
                  <category.icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white leading-tight pt-2">
                  {category.title}
                </h3>
              </div>

              <div className="flex flex-wrap gap-2">
                {category.skills.map((skill, skillIndex) => (
                  <span
                    key={skillIndex}
                    className="px-3 py-1.5 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-full text-sm font-medium border border-slate-200 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-slate-600 transition-colors"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
