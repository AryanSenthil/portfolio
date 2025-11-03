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

  return (
    <section id="publications" className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            Publications
          </h2>
          <div className="w-20 h-1 bg-blue-600 mx-auto rounded-full mb-6"></div>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Research contributions and academic publications
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
              className="bg-slate-50 rounded-2xl p-8 hover:bg-blue-50 transition-all border border-slate-200 hover:border-blue-200 group"
            >
              <div className="flex items-start gap-4">
                <div className="p-3 bg-white rounded-xl shadow-sm group-hover:bg-blue-100 transition-colors">
                  <FileCheck className="w-6 h-6 text-blue-600" />
                </div>
                
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-900 mb-2 group-hover:text-blue-600 transition-colors">
                    {pub.title}
                  </h3>
                  <p className="text-slate-600 mb-2">{pub.authors}</p>
                  <p className="text-sm text-slate-500 mb-4">
                    <span className="font-medium text-blue-600">{pub.venue}</span> • {pub.year}
                  </p>
                  
                  {pub.abstract && (
                    <p className="text-slate-600 text-sm leading-relaxed mb-4">
                      {pub.abstract}
                    </p>
                  )}

                  <div className="flex flex-wrap gap-3">
                    {pub.url && (
                      <Button
                        variant="outline"
                        size="sm"
                        asChild
                        className="text-blue-600 border-blue-200 hover:bg-blue-50"
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
                        className="text-slate-600 hover:bg-slate-100"
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
              <p className="text-slate-500 mb-4">No publications yet.</p>
              <p className="text-sm text-slate-400">
                Add publications through the dashboard.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}