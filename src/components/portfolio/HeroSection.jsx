import React, { useState } from 'react';
import { User, Download, Mail, Linkedin, Github, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import CVViewer from './CVViewer';

export default function HeroSection() {
  const [showCV, setShowCV] = useState(false);

  return (
    <section id="home" className="min-h-screen flex items-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-slate-800 pt-20 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-6 py-20">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Left: Portrait */}
          <div className="flex flex-col items-center md:items-end gap-6">
            <div className="bg-slate-100 dark:bg-slate-700 rounded-3xl overflow-hidden w-[36rem] h-[42rem] flex items-center justify-center shadow-2xl dark:shadow-slate-900/50">
              <img src="/portfolio/profile.jpg" alt="Aryan Yamini Senthil" className="w-full h-full object-cover" />
            </div>
            
            {/* Social Icons */}
            <div className="flex items-center gap-3">
              <a
                href="mailto:aryanyaminisenthil@gmail.com"
                className="p-3 bg-slate-700 dark:bg-slate-600 hover:bg-blue-600 dark:hover:bg-blue-600 rounded-lg transition-all"
                aria-label="Email"
              >
                <Mail className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://www.linkedin.com/in/aryan-yamini-senthil-18125b243"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 dark:bg-slate-600 hover:bg-blue-600 dark:hover:bg-blue-600 rounded-lg transition-all"
                aria-label="LinkedIn"
              >
                <Linkedin className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://github.com/AryanSenthil"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 dark:bg-slate-600 hover:bg-blue-600 dark:hover:bg-blue-600 rounded-lg transition-all"
                aria-label="GitHub"
              >
                <Github className="w-5 h-5 text-white" />
              </a>
            </div>
          </div>

          {/* Right: Bio */}
          <div className="space-y-6">
            <div className="space-y-2">
              <h1 className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-white leading-tight">
                Aryan Yamini Senthil
              </h1>
              <p className="text-2xl text-blue-600 dark:text-blue-400 font-medium">
                Undergraduate Researcher and PhD Candidate
              </p>
            </div>
            
            <div className="h-1 w-20 bg-blue-600 dark:bg-blue-500 rounded-full"></div>

            <p className="text-lg text-slate-600 dark:text-slate-300 leading-relaxed max-w-xl">
              I am a graduating senior passionate about advancing research in deep learning and autonomous systems.
              With a strong foundation in applied machine learning, sensor design, and algorithm development, I'm seeking PhD opportunities
              to contribute to cutting-edge research and make meaningful impact in academia and industry.
            </p>

            {/* Research Interests - Isolated Section */}
            <div className="bg-blue-50 dark:bg-slate-800/50 border-l-4 border-blue-600 dark:border-blue-500 rounded-lg p-6 max-w-xl shadow-md">
              <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-4">
                Research Interests
              </h3>
              <div className="flex flex-wrap gap-3">
                <span className="px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-full text-sm font-medium shadow-sm">
                  Deep Learning
                </span>
                <span className="px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-full text-sm font-medium shadow-sm">
                  Reinforcement Learning
                </span>
                <span className="px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-full text-sm font-medium shadow-sm">
                  Multimodal AI
                </span>
              </div>
            </div>

            <div className="flex gap-4 pt-4">
              <Button
                className="px-8 py-6 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl"
                onClick={() => setShowCV(true)}
              >
                <Eye className="w-5 h-5 mr-2" />
                View CV
              </Button>
              <Button
                variant="outline"
                className="px-6 py-6 rounded-xl border-2 border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-700 dark:bg-slate-800 dark:text-slate-200 transition-all shadow-lg hover:shadow-xl"
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = '/portfolio/cv.pdf';
                  link.download = 'Aryan_Yamini_Senthil_CV.pdf';
                  link.click();
                }}
              >
                <Download className="w-6 h-6" />
              </Button>
              <a 
                href="#contact" 
                className="px-8 py-3 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-xl font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-all border-2 border-slate-200 dark:border-slate-600 flex items-center"
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById('contact').scrollIntoView({ behavior: 'smooth' });
                }}
              >
                Get in Touch
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* CV Viewer Modal */}
      {showCV && (
        <CVViewer
          onClose={() => setShowCV(false)}
        />
      )}
    </section>
  );
}