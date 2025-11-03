import React from 'react';
import { motion } from 'framer-motion';
import { GraduationCap, Award, BookOpen, Target } from 'lucide-react';

export default function AboutSection() {
  const highlights = [
    {
      icon: GraduationCap,
      title: 'Education',
      description: 'Senior at [University Name]',
      detail: 'Major in [Your Major] • GPA: X.XX'
    },
    {
      icon: Award,
      title: 'Achievements',
      description: 'Academic excellence & recognition',
      detail: 'Dean\'s List • Research Awards'
    },
    {
      icon: BookOpen,
      title: 'Publications',
      description: 'Contributing to the field',
      detail: 'Conference papers & journals'
    },
    {
      icon: Target,
      title: 'Goals',
      description: 'Pursuing PhD opportunities',
      detail: 'Focus on [Research Area]'
    }
  ];

  return (
    <section id="about" className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            About Me
          </h2>
          <div className="w-20 h-1 bg-blue-600 mx-auto rounded-full"></div>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {highlights.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-slate-50 rounded-2xl p-6 hover:bg-blue-50 transition-all duration-300 group"
            >
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4 group-hover:bg-blue-600 transition-colors">
                <item.icon className="w-6 h-6 text-blue-600 group-hover:text-white transition-colors" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2">{item.title}</h3>
              <p className="text-slate-600 text-sm mb-1">{item.description}</p>
              <p className="text-slate-500 text-xs">{item.detail}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-16 bg-gradient-to-br from-blue-50 to-slate-50 rounded-3xl p-8 md:p-12"
        >
          <h3 className="text-2xl font-bold text-slate-900 mb-6">Academic Background</h3>
          <div className="space-y-4 text-slate-600 leading-relaxed">
            <p>
              Throughout my undergraduate journey, I've cultivated a deep passion for [Your Field]. 
              My coursework has provided me with a strong theoretical foundation, while research 
              experiences have honed my practical skills in [Specific Skills/Methods].
            </p>
            <p>
              I've had the privilege of working with [Professors/Labs/Organizations], where I've 
              contributed to [Brief Description of Work]. These experiences have reinforced my 
              commitment to pursuing advanced research and making meaningful contributions to the field.
            </p>
            <p>
              As I prepare to graduate, I'm eager to continue this journey at the doctoral level, 
              where I can delve deeper into [Specific Research Interests] and collaborate with 
              leading researchers in the field.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}