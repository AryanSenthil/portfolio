import React from 'react';
import { Mail, Linkedin, Github } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-slate-900 dark:bg-slate-950 text-white py-12">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-center md:text-left">
            <h3 className="text-xl font-bold mb-2 dark:text-white">Aryan Yamini Senthil</h3>
            <p className="text-slate-400 dark:text-slate-300 text-sm">
              Undergraduate Researcher | PhD Candidate | Engineering Physics
            </p>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="mailto:aryanyaminisenthil@gmail.com"
              className="p-3 bg-slate-800 dark:bg-slate-900 rounded-lg hover:bg-blue-600 dark:hover:bg-blue-500 transition-all"
              aria-label="Email"
            >
              <Mail className="w-5 h-5" />
            </a>
            <a
              href="https://www.linkedin.com/in/aryan-yamini-senthil-18125b243"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 bg-slate-800 dark:bg-slate-900 rounded-lg hover:bg-blue-600 dark:hover:bg-blue-500 transition-all"
              aria-label="LinkedIn"
            >
              <Linkedin className="w-5 h-5" />
            </a>
            <a
              href="https://github.com/AryanSenthil"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 bg-slate-800 dark:bg-slate-900 rounded-lg hover:bg-blue-600 dark:hover:bg-blue-500 transition-all"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-slate-800 dark:border-slate-700 text-center text-slate-400 dark:text-slate-300 text-sm">
          <p>© {new Date().getFullYear()} Aryan Yamini Senthil. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}