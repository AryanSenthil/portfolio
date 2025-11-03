import React from 'react';
import { User, Download, Mail, Linkedin, Github, Twitter, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function HeroSection() {
  return (
    <section id="home" className="min-h-screen flex items-center bg-gradient-to-br from-slate-50 to-blue-50/30 pt-20">
      <div className="max-w-7xl mx-auto px-6 py-20">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Left: Portrait */}
          <div className="flex flex-col items-center md:items-end gap-6">
            <div className="bg-slate-100 rounded-3xl overflow-hidden w-[36rem] h-[42rem] flex items-center justify-center shadow-2xl">
              <img src="/profile.jpg" alt="Aryan Yamini Senthil" className="w-full h-full object-cover" />
            </div>
            
            {/* Social Icons */}
            <div className="flex items-center gap-3">
              <a
                href="mailto:aryanyaminisenthil@gmail.com"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="Email"
              >
                <Mail className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://x.com/AryanSenthil"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="Twitter"
              >
                <Twitter className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://www.linkedin.com/in/aryan-yamini-senthil-18125b243"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="LinkedIn"
              >
                <Linkedin className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://github.com/AryanSenthil"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="GitHub"
              >
                <Github className="w-5 h-5 text-white" />
              </a>
            </div>
          </div>

          {/* Right: Bio */}
          <div className="space-y-6">
            <div className="space-y-2">
              <h1 className="text-5xl md:text-6xl font-bold text-slate-900 leading-tight">
                Aryan Yamini Senthil
              </h1>
              <p className="text-2xl text-blue-600 font-medium">
                Senior Undergraduate Researcher and PhD Candidate
              </p>
            </div>
            
            <div className="h-1 w-20 bg-blue-600 rounded-full"></div>
            
            <p className="text-lg text-slate-600 leading-relaxed max-w-xl">
              I am a graduating senior passionate about advancing research in deep learning and autonomous systems.
              With a strong foundation in applied machine learning, sensor design, and algorithm development, I'm seeking PhD opportunities
              to contribute to cutting-edge research and make meaningful impact in academia and industry.
            </p>

            <p className="text-base text-slate-600 leading-relaxed max-w-xl">
              My research interests span deep learning, reinforcement learning (agents), and multimodal AI,
              where I strive to bridge theoretical foundations with practical applications.
            </p>

            <div className="flex gap-4 pt-4">
              <Button
                className="px-8 py-6 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl"
                onClick={() => {
                  window.open('/cv.pdf', '_blank');
                }}
              >
                <Eye className="w-5 h-5 mr-2" />
                View CV
              </Button>
              <Button
                variant="outline"
                className="px-6 py-6 rounded-xl border-2 border-slate-300 hover:bg-slate-50 transition-all shadow-lg hover:shadow-xl"
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = '/cv.pdf';
                  link.download = 'Aryan_Yamini_Senthil_CV.pdf';
                  link.click();
                }}
              >
                <Download className="w-6 h-6" />
              </Button>
              <a 
                href="#contact" 
                className="px-8 py-3 bg-white text-slate-700 rounded-xl font-medium hover:bg-slate-50 transition-all border-2 border-slate-200 flex items-center"
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
    </section>
  );
}