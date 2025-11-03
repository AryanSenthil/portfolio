import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { base44 } from '@/api/base44Client';
import { motion } from 'framer-motion';

import Navigation from '../components/portfolio/Navigation';
import ScrollProgress from '../components/portfolio/ScrollProgress';
import HeroSection from '../components/portfolio/HeroSection';
import AboutSection from '../components/portfolio/AboutSection';
import ExperienceSection from '../components/portfolio/ExperienceSection';
import PublicationsSection from '../components/portfolio/PublicationsSection';
import SkillsSection from '../components/portfolio/SkillsSection';
import InvolvementSection from '../components/portfolio/InvolvementSection';
import PortfolioCard from '../components/portfolio/PortfolioCard';
import HTMLViewer from '../components/portfolio/HTMLViewer';
import ContactSection from '../components/portfolio/ContactSection';
import Footer from '../components/portfolio/Footer';

export default function Portfolio() {
  const [activeSection, setActiveSection] = useState('home');
  const [selectedItem, setSelectedItem] = useState(null);

  const { data: portfolioItems = [] } = useQuery({
    queryKey: ['portfolioItems'],
    queryFn: () => base44.entities.PortfolioItem.list('order'),
    initialData: []
  });

  const researchItems = portfolioItems.filter(item => item.type === 'research');
  const problemItems = portfolioItems.filter(item => item.type === 'problem');

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['home', 'about', 'experience', 'publications', 'research', 'problems', 'skills', 'involvement', 'contact'];
      const scrollPosition = window.scrollY + 100;

      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (scrollPosition >= offsetTop && scrollPosition < offsetTop + offsetHeight) {
            setActiveSection(section);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') setSelectedItem(null);
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, []);

  return (
    <div className="min-h-screen bg-white dark:bg-slate-900 transition-colors duration-200">
      <ScrollProgress />
      <Navigation activeSection={activeSection} />

      <HeroSection />
      
      <AboutSection />

      <ExperienceSection />

      <PublicationsSection />

      {/* Research Section */}
      <section id="research" className="py-24 bg-white dark:bg-slate-900">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
              Research
            </h2>
            <div className="w-20 h-1 bg-blue-600 dark:bg-blue-500 mx-auto rounded-full mb-6"></div>
            <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
              Explore my research projects and contributions to the field
            </p>
          </motion.div>

          {researchItems.length > 0 ? (
            <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
              {researchItems.map((item, index) => (
                <PortfolioCard
                  key={item.id}
                  item={item}
                  index={index}
                  onClick={() => setSelectedItem(item)}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-slate-500 dark:text-slate-400 mb-4">No research items yet.</p>
              <p className="text-sm text-slate-400 dark:text-slate-500">
                Add research items through the dashboard to showcase your work.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Projects Section */}
      <section id="problems" className="py-24 bg-slate-50 dark:bg-slate-800">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
              Projects
            </h2>
            <div className="w-20 h-1 bg-blue-600 dark:bg-blue-500 mx-auto rounded-full mb-6"></div>
            <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
              Innovative solutions to complex challenges
            </p>
          </motion.div>

          {problemItems.length > 0 ? (
            <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
              {problemItems.map((item, index) => (
                <PortfolioCard
                  key={item.id}
                  item={item}
                  index={index}
                  onClick={() => setSelectedItem(item)}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-slate-500 dark:text-slate-400 mb-4">No problem solutions yet.</p>
              <p className="text-sm text-slate-400 dark:text-slate-500">
                Add problem solutions through the dashboard to showcase your achievements.
              </p>
            </div>
          )}
        </div>
      </section>

      <SkillsSection />

      <InvolvementSection />

      <ContactSection />
      
      <Footer />

      {/* HTML Content Viewer Modal */}
      {selectedItem && (
        <HTMLViewer
          item={selectedItem}
          onClose={() => setSelectedItem(null)}
        />
      )}
    </div>
  );
}