// Mock base44 API client for standalone demo

const mockPortfolioItems = [
  {
    id: '1',
    title: 'Self Powered Onion-Peel Pressure Sensor',
    type: 'research',
    description: 'Deep Learning-Based Load Characterization and Deformation Prediction for Bio-Sourced Onion Peel Piezoelectric Sensors',
    image_url: '/portfolio/projects/onion_sensor/images/Onion Ink/sensor.png',
    html_file_url: '/portfolio/projects/onion_sensor/src/onion_sensor.html',
    order: 1
  },
  {
    id: '2',
    title: 'AI-Driven Structural Health Monitoring',
    type: 'research',
    description: 'Damage Detection in Carbon Fiber Composities using Deep Learning enabled Flexoelectric Sensors',
    image_url: '/portfolio/projects/Damage/images/Slide1.png',
    html_file_url: '/portfolio/projects/Damage/damage.html',
    order: 2
  },
  {
    id: '3',
    title: 'Contact Detection Bot',
    type: 'problem',
    description: 'Automated Contact Detection for Instron Compression Testing of Piezoelectric Pressure Sensors',
    html_file_url: '/portfolio/projects/Contact-Detection/instron_extension.html',
    order: 1
  },
  {
    id: '4',
    title: 'Laser Coordinate Tracker',
    type: 'problem',
    description: 'Computer Vision-Based Laser Position Measurement in Torsion Balance Experiments',
    html_file_url: '/portfolio/projects/Cavendish/cavendish.html',
    order: 2
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
    end_date: 'Aug 2025',
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
    venue: 'The American Society for Composites (ASC), Volume 4, San Diego, CA',
    year: 2024,
    url: 'https://doi.org/10.1007/978-3-032-05216-2_20',
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
