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
    title: 'Senior Software Engineer',
    company: 'Tech Company',
    location: 'Remote',
    startDate: '2022-01-01',
    endDate: null,
    current: true,
    description: 'Leading development of innovative software solutions.',
    order: 1
  },
  {
    id: '2',
    title: 'Software Engineer',
    company: 'Previous Company',
    location: 'San Francisco, CA',
    startDate: '2020-01-01',
    endDate: '2021-12-31',
    current: false,
    description: 'Developed web applications and APIs.',
    order: 2
  }
];

const mockPublications = [
  {
    id: '1',
    title: 'Sample Publication Title',
    authors: 'Author Name, Co-Author Name',
    venue: 'International Conference on Example Topics',
    year: 2023,
    url: 'https://example.com',
    abstract: 'This is a sample abstract for a publication.',
    order: 1
  },
  {
    id: '2',
    title: 'Another Research Paper',
    authors: 'Author Name, et al.',
    venue: 'Journal of Example Research',
    year: 2022,
    url: 'https://example.com',
    abstract: 'This is another sample abstract.',
    order: 2
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
