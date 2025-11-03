// Mock base44 API client for standalone demo

const mockPortfolioItems = [
  {
    id: '1',
    title: 'Sample Research Project',
    type: 'research',
    description: 'This is a sample research project demonstrating portfolio functionality.',
    image_url: 'https://via.placeholder.com/600x400',
    content: '<h1>Research Project</h1><p>This is sample content for the research project.</p>',
    order: 1
  },
  {
    id: '2',
    title: 'Another Research Study',
    type: 'research',
    description: 'Another example of research work.',
    image_url: 'https://via.placeholder.com/600x400',
    content: '<h1>Research Study</h1><p>This is sample content for the research study.</p>',
    order: 2
  },
  {
    id: '3',
    title: 'Problem: Data Processing Challenge',
    type: 'problem',
    description: 'Solved a complex data processing challenge.',
    image_url: 'https://via.placeholder.com/600x400',
    content: '<h1>Data Processing Solution</h1><p>Details about the problem and solution.</p>',
    order: 1
  }
];

const mockExperiences = [
  {
    id: '1',
    position: 'Undergraduate Research Assistant',
    company: 'University of Oklahoma',
    location: 'Norman, OK',
    start_date: 'May 2022',
    end_date: 'Present',
    current: true,
    description: [
      'Published paper on flexoelectric sensors for damage detection',
      'Designed and 3D printed ultrasensitive smart sensors',
      'Developed self-powered AI-driven sensing technologies',
      'Built novel deep learning models that allow intelligent pressure sensors',
      'Created neural network models for composite damage localization'
    ],
    order: 1
  },
  {
    id: '2',
    position: 'CEO and Cofounder',
    company: 'Dirac Technologies',
    location: 'Norman, OK',
    start_date: 'May 2024',
    end_date: 'Jun 2025',
    current: false,
    description: [
      'Secured funding for autonomous robotic arms development',
      'Built imitation learning framework for manipulation tasks',
      'Integrated object detection and teleoperation systems',
      'Led 5-member team across hardware and AI'
    ],
    order: 2
  },
  {
    id: '3',
    position: 'Math Tutor',
    company: 'Math Center - University of Oklahoma',
    location: 'Norman, OK',
    start_date: 'Oct 2022',
    end_date: 'May 2023',
    current: false,
    description: [
      'Tutored Calculus, Differential Equations, and Linear Algebra',
      'Guided students through exams and homework'
    ],
    order: 3
  },
  {
    id: '4',
    position: 'Community Service Chair',
    company: 'Pi Kappa Phi Fraternity - University of Oklahoma',
    location: 'Norman, OK',
    start_date: 'Aug 2022',
    end_date: 'Dec 2023',
    current: false,
    description: [
      'Organized national philanthropy initiative for people with disabilities',
      'Coordinated fundraising and volunteer events',
      'Led participation in university\'s largest service projects'
    ],
    order: 4
  },
  {
    id: '5',
    position: 'Executive Board Member',
    company: 'OU Cousins - University of Oklahoma',
    location: 'Norman, OK',
    start_date: 'Sep 2020',
    end_date: 'May 2021',
    current: false,
    description: [
      'Facilitated intercultural programs for international students',
      'Promoted diversity and inclusion across campus'
    ],
    order: 5
  }
];

const mockPublications = [
  {
    id: '1',
    title: 'Design of Flexible and Ultrasensitive 3D Printed Flexoelectric Sensor for Self-Powered Damage Detection of Composite Structures',
    authors: 'Kuntal Maity, Mrinal Saha, Aryan Senthil, Anirban Mondal',
    venue: '39th Annual Technical Conference of the American Society for Composites (ASC) [In Press]',
    year: 2024,
    url: 'https://sites.google.com/view/2024asc-sandiego/home',
    order: 1
  }
];

export const base44 = {
  entities: {
    PortfolioItem: {
      list: (sortBy = 'order') => Promise.resolve(mockPortfolioItems)
    },
    Experience: {
      list: (sortBy = 'order') => Promise.resolve(mockExperiences)
    },
    Publication: {
      list: (sortBy = 'order') => Promise.resolve(mockPublications)
    }
  }
};
