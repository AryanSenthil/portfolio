import React, { useState } from 'react';
import { User, Download, Mail, Linkedin, Github, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import CVViewer from './CVViewer';

export default function HeroSection() {
  const [showCV, setShowCV] = useState(false);

  return (
    <section id="home" className="min-h-screen flex items-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-slate-800 pt-20 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 sm:py-20">
        <div className="grid md:grid-cols-2 gap-8 md:gap-12 items-center">
          {/* Left: Portrait */}
          <div className="flex flex-col items-center md:items-end gap-6 order-1 md:order-none">
            <div className="bg-slate-100 dark:bg-slate-700 rounded-3xl overflow-hidden w-full max-w-sm md:max-w-md lg:max-w-lg aspect-[3/3.5] flex items-center justify-center shadow-2xl dark:shadow-slate-900/50">
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
          <div className="space-y-4 sm:space-y-6 order-2 md:order-none">
            <div className="space-y-2 text-center md:text-left">
              <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-slate-900 dark:text-white leading-tight">
                Aryan Yamini Senthil
              </h1>
              <p className="text-lg sm:text-xl md:text-2xl text-blue-600 dark:text-blue-400 font-medium">
                Engineering Physics Graduate | Seeking Opportunities in Aerospace, Materials & AI
              </p>
            </div>

            <div className="h-1 w-20 bg-blue-600 dark:bg-blue-500 rounded-full mx-auto md:mx-0"></div>

            <p className="text-base sm:text-lg text-slate-600 dark:text-slate-300 leading-relaxed max-w-xl text-center md:text-left">
              I am an Engineering Physics graduate with a cross-disciplinary background in physics and aerospace engineering.
              With a strong foundation in machine learning, materials, and computer-aided design, I'm seeking job opportunities
              to contribute to cutting-edge engineering and make meaningful impact in industry.
            </p>

            <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 pt-4 items-stretch sm:items-center">
              <Button
                className="px-6 sm:px-8 py-4 sm:py-6 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl text-sm sm:text-base"
                onClick={() => setShowCV(true)}
              >
                <Eye className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                View Resume
              </Button>
              <div className="flex gap-3 sm:gap-4">
                <Button
                  variant="outline"
                  className="flex-1 sm:flex-none px-4 sm:px-6 py-4 sm:py-6 rounded-xl border-2 border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-700 dark:bg-slate-800 dark:text-slate-200 transition-all shadow-lg hover:shadow-xl"
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = '/portfolio/aryansenthilresume.pdf';
                    link.download = 'Aryan_Senthil_Resume.pdf';
                    link.click();
                  }}
                >
                  <Download className="w-5 h-5 sm:w-6 sm:h-6" />
                </Button>
                <a
                  href="#contact"
                  className="flex-1 sm:flex-none px-6 sm:px-8 py-3 sm:py-3 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-xl font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-all border-2 border-slate-200 dark:border-slate-600 flex items-center justify-center text-sm sm:text-base"
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
      </div>

      {/* Resume Viewer Modal */}
      {showCV && (
        <CVViewer
          onClose={() => setShowCV(false)}
        />
      )}
    </section>
  );
}