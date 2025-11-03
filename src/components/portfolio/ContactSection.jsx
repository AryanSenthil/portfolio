import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Linkedin, Send, CheckCircle, Phone, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { easings, hoverLift, iconRotate } from '@/utils/motionConfig';

export default function ContactSection() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [error, setError] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(false);

    try {
      const response = await fetch('https://formspree.io/f/xpwoednk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          subject: formData.subject,
          message: formData.message,
        }),
      });

      if (response.ok) {
        setIsSubmitted(true);
        setFormData({ name: '', email: '', subject: '', message: '' });
        setTimeout(() => setIsSubmitted(false), 5000);
      } else {
        setError(true);
        setTimeout(() => setError(false), 5000);
      }
    } catch (error) {
      console.error('Error sending email:', error);
      setError(true);
      setTimeout(() => setError(false), 5000);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section id="contact" className="py-16 sm:py-24 bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            Get in Touch
          </h2>
          <motion.div
            initial={{ width: 0 }}
            whileInView={{ width: 80 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            viewport={{ once: true }}
            className="h-1 bg-gradient-to-r from-blue-600 to-blue-400 dark:from-blue-500 dark:to-blue-400 mx-auto rounded-full mb-6"
          ></motion.div>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            I'm actively seeking PhD opportunities and would love to hear from professors,
            recruiters, and collaborators interested in my work.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
          {/* Contact Info */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            <div>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">Connect With Me</h3>
              <p className="text-slate-600 dark:text-slate-300 mb-8">
                Feel free to reach out directly via email or connect with me on LinkedIn.
                I typically respond within 24 hours.
              </p>
            </div>

            <div className="space-y-4">
              <motion.a
                href="mailto:aryanyaminisenthil@gmail.com"
                whileHover="hover"
                variants={hoverLift}
                className="flex items-center gap-4 p-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-xl hover:bg-blue-50/80 dark:hover:bg-slate-750/80 transition-all duration-400 group border border-slate-200/50 dark:border-slate-700/50 hover:border-blue-200 dark:hover:border-blue-400 shadow-md hover:shadow-float dark:shadow-glass relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-50/20 via-transparent to-white/10 dark:from-blue-900/10 dark:to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
                <motion.div
                  variants={iconRotate}
                  className="relative z-10 p-3 bg-blue-50/80 dark:bg-blue-900/50 backdrop-blur-sm rounded-lg group-hover:bg-blue-100 dark:group-hover:bg-blue-800/70 transition-all duration-300"
                >
                  <Mail className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </motion.div>
                <div className="relative z-10">
                  <p className="font-medium text-slate-900 dark:text-white">Email</p>
                  <p className="text-sm text-slate-600 dark:text-slate-300">aryanyaminisenthil@gmail.com</p>
                </div>
              </motion.a>

              <motion.a
                href="tel:+14054147622"
                whileHover="hover"
                variants={hoverLift}
                className="flex items-center gap-4 p-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-xl hover:bg-blue-50/80 dark:hover:bg-slate-750/80 transition-all duration-400 group border border-slate-200/50 dark:border-slate-700/50 hover:border-blue-200 dark:hover:border-blue-400 shadow-md hover:shadow-float dark:shadow-glass relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-50/20 via-transparent to-white/10 dark:from-blue-900/10 dark:to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
                <motion.div
                  variants={iconRotate}
                  className="relative z-10 p-3 bg-blue-50/80 dark:bg-blue-900/50 backdrop-blur-sm rounded-lg group-hover:bg-blue-100 dark:group-hover:bg-blue-800/70 transition-all duration-300"
                >
                  <Phone className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </motion.div>
                <div className="relative z-10">
                  <p className="font-medium text-slate-900 dark:text-white">Phone</p>
                  <p className="text-sm text-slate-600 dark:text-slate-300">(405) 414-7622</p>
                </div>
              </motion.a>

              <motion.a
                href="https://www.linkedin.com/in/aryan-yamini-senthil-18125b243"
                target="_blank"
                rel="noopener noreferrer"
                whileHover="hover"
                variants={hoverLift}
                className="flex items-center gap-4 p-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-xl hover:bg-blue-50/80 dark:hover:bg-slate-750/80 transition-all duration-400 group border border-slate-200/50 dark:border-slate-700/50 hover:border-blue-200 dark:hover:border-blue-400 shadow-md hover:shadow-float dark:shadow-glass relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-50/20 via-transparent to-white/10 dark:from-blue-900/10 dark:to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
                <motion.div
                  variants={iconRotate}
                  className="relative z-10 p-3 bg-blue-50/80 dark:bg-blue-900/50 backdrop-blur-sm rounded-lg group-hover:bg-blue-100 dark:group-hover:bg-blue-800/70 transition-all duration-300"
                >
                  <Linkedin className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </motion.div>
                <div className="relative z-10">
                  <p className="font-medium text-slate-900 dark:text-white">LinkedIn</p>
                  <p className="text-sm text-slate-600 dark:text-slate-300">Connect professionally</p>
                </div>
              </motion.a>
            </div>
          </motion.div>

          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, ease: easings.emphasis }}
            viewport={{ once: true }}
          >
            <form onSubmit={handleSubmit} className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-glass rounded-2xl p-8 shadow-float dark:shadow-glass border border-slate-200/50 dark:border-slate-700/50 relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-50/10 via-transparent to-white/5 dark:from-blue-900/5 dark:to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none"></div>
              <div className="space-y-5 relative z-10">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Your Name
                  </label>
                  <Input
                    required
                    name="name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="John Doe"
                    className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Your Email
                  </label>
                  <Input
                    required
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    placeholder="john@example.com"
                    className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Subject
                  </label>
                  <Input
                    required
                    name="subject"
                    value={formData.subject}
                    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                    placeholder="PhD Opportunity Inquiry"
                    className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Message
                  </label>
                  <Textarea
                    required
                    name="message"
                    value={formData.message}
                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                    placeholder="Tell me about your research interests or opportunities..."
                    rows={5}
                    className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 resize-none dark:text-white"
                  />
                </div>

                <Button
                  type="submit"
                  disabled={isSubmitting || isSubmitted}
                  className={`w-full py-6 rounded-xl font-medium transition-colors ${
                    error
                      ? 'bg-red-600 hover:bg-red-700'
                      : isSubmitted
                      ? 'bg-green-600 hover:bg-green-700'
                      : 'bg-blue-600 hover:bg-blue-700'
                  } text-white`}
                >
                  {error ? (
                    <>
                      <AlertCircle className="w-5 h-5 mr-2" />
                      Failed to Send
                    </>
                  ) : isSubmitted ? (
                    <>
                      <CheckCircle className="w-5 h-5 mr-2" />
                      Message Sent!
                    </>
                  ) : isSubmitting ? (
                    'Sending...'
                  ) : (
                    <>
                      <Send className="w-5 h-5 mr-2" />
                      Send Message
                    </>
                  )}
                </Button>
              </div>
            </form>
          </motion.div>
        </div>
      </div>
    </section>
  );
}