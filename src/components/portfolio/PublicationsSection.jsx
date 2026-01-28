import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { base44 } from '@/api/base44Client';
import { FileText, ExternalLink, FileCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function PublicationsSection() {
  const { data: publications = [] } = useQuery({
    queryKey: ['publications'],
    queryFn: () => base44.entities.Publication.list('-year'),
    initialData: []
  });

  const renderAuthors = (authorsString) => {
    const parts = authorsString.split('Aryan Senthil');
    if (parts.length === 1) return authorsString;

    return (
      <>
        {parts[0]}
        <strong>Aryan Senthil</strong>
        {parts[1]}
      </>
    );
  };

  return (
    <section id="publications" className="py-16 sm:py-24 bg-white dark:bg-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            Publications
          </h2>
          <div className="w-20 h-1 bg-blue-600 dark:bg-blue-400 mx-auto rounded-full mb-6"></div>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Published work and academic contributions
          </p>
        </motion.div>

        <div className="max-w-5xl mx-auto space-y-6">
          {publications.map((pub, index) => (
            <motion.div
              key={pub.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-slate-50 dark:bg-slate-800 rounded-2xl p-8 hover:bg-blue-50 dark:hover:bg-slate-750 transition-all border border-slate-200 dark:border-slate-700 hover:border-blue-200 dark:hover:border-blue-400 group"
            >
              <div className="flex items-start gap-4">
                <div className="p-3 bg-white dark:bg-slate-900 rounded-xl shadow-sm group-hover:bg-blue-100 dark:group-hover:bg-blue-900 transition-colors">
                  <FileCheck className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>

                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {pub.title}
                  </h3>
                  <p className="text-slate-600 dark:text-slate-300 mb-2">{renderAuthors(pub.authors)}</p>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                    <span className="font-medium text-blue-600 dark:text-blue-400">{pub.venue}</span> • {pub.year}
                  </p>

                  {pub.abstract && (
                    <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed mb-4">
                      {pub.abstract}
                    </p>
                  )}

                  <div className="flex flex-wrap gap-3">
                    {pub.url && (
                      <Button
                        variant="outline"
                        size="sm"
                        asChild
                        className="text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900"
                      >
                        <a href={pub.url} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="w-4 h-4 mr-2" />
                          View Paper
                        </a>
                      </Button>
                    )}
                    {pub.pdf_url && (
                      <Button
                        variant="outline"
                        size="sm"
                        asChild
                        className="text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                      >
                        <a href={pub.pdf_url} target="_blank" rel="noopener noreferrer">
                          <FileText className="w-4 h-4 mr-2" />
                          PDF
                        </a>
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}

          {publications.length === 0 && (
            <div className="text-center py-12">
              <p className="text-slate-500 dark:text-slate-400 mb-4">No publications yet.</p>
              <p className="text-sm text-slate-400 dark:text-slate-500">
                Add publications through the dashboard.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}