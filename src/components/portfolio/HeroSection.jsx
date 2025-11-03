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
            <div className="bg-slate-100 rounded-3xl overflow-hidden w-96 h-96 flex items-center justify-center shadow-2xl">
              <User className="w-40 h-40 text-slate-400" />
              <div className="absolute inset-0 bg-gradient-to-t from-slate-900/20 to-transparent"></div>
            </div>
            <p className="text-xs text-center text-slate-500">
              Replace this placeholder with your photo
            </p>
            
            {/* Social Icons */}
            <div className="flex items-center gap-3">
              <a
                href="mailto:your.email@example.com"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="Email"
              >
                <Mail className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://twitter.com/yourhandle"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="Twitter"
              >
                <Twitter className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://linkedin.com/in/yourprofile"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 bg-slate-700 hover:bg-blue-600 rounded-lg transition-all"
                aria-label="LinkedIn"
              >
                <Linkedin className="w-5 h-5 text-white" />
              </a>
              <a
                href="https://github.com/yourusername"
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
                Your Name
              </h1>
              <p className="text-2xl text-blue-600 font-medium">
                PhD Candidate | Researcher
              </p>
            </div>
            
            <div className="h-1 w-20 bg-blue-600 rounded-full"></div>
            
            <p className="text-lg text-slate-600 leading-relaxed max-w-xl">
              I am a graduating senior passionate about advancing research in [Your Field]. 
              With a strong foundation in [Your Expertise], I'm seeking PhD opportunities 
              to contribute to cutting-edge research and make meaningful impact in academia 
              and industry.
            </p>

            <p className="text-base text-slate-600 leading-relaxed max-w-xl">
              My research interests span [Interest 1], [Interest 2], and [Interest 3], 
              where I strive to bridge theoretical foundations with practical applications.
            </p>

            <div className="flex gap-4 pt-4">
              <Button
                className="px-8 py-6 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl"
                onClick={() => {
                  // Replace with your CV file URL
                  window.open('/path-to-your-cv.pdf', '_blank');
                }}
              >
                <Eye className="w-5 h-5 mr-2" />
                View CV
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="px-6 py-6 rounded-xl border-2 border-slate-300 hover:bg-slate-50 transition-all shadow-lg"
                onClick={() => {
                  // Replace with your CV file URL for download
                  const link = document.createElement('a');
                  link.href = '/path-to-your-cv.pdf';
                  link.download = 'Your_Name_CV.pdf';
                  link.click();
                }}
              >
                <Download className="w-5 h-5" />
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