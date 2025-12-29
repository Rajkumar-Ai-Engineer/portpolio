// Professional AI Portfolio Project Database
// 5 Featured Projects

const projectDatabase = {
    all: [
        {
            id: 'project-1',
            title: 'Real-Time Fake News Detection using Agentic AI System',
            category: 'Agentic AI',
            domain: 'Information Verification',
            description: 'An advanced agentic AI system that demonstrates using LangGraph to verify news authenticity in real-time. The system leverages real-time search tools and RAG (Retrieval-Augmented Generation) to reduce hallucination, providing evidence-based verification with caching for improved performance.',
            image: 'videos/1.mp4',
            video: 'videos/1.mp4',
            technologies: ['LangGraph', 'Tavily News Tool', 'RAG', 'Pinecone', 'LangChain', 'Groq', 'Llama Models', 'MCP'],
            frameworks: ['LangGraph', 'LangChain', 'Pinecone'],
            keyFeatures: [
                'Real-time fake news detection agent',
                'News search agent for comprehensive verification',
                'RAG integration to reduce AI hallucination',
                'Evidence-based verification with caching',
                'Multi-agent collaboration system',
                'MCP tools integration for enhanced functionality'
            ],
            technicalDetails: {
                architecture: 'Multi-Agent System with LangGraph',
                agents: '2 specialized agents (Detection + Search)',
                llm: 'Groq with Llama models',
                vectorDB: 'Pinecone for RAG',
                searchTool: 'Tavily News API',
                framework: 'LangChain + LangGraph'
            },
            applications: [
                'Social media fact-checking',
                'News verification platforms',
                'Misinformation detection systems',
                'Journalism support tools'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/fake-news-detection',
            demoLink: '',
            tags: ['Agentic AI', 'LangGraph', 'Fake News Detection', 'RAG', 'Multi-Agent System', 'Real-time Verification'],
            featured: true,
            projectNumber: 1,
            totalProjects: 5
        },
        {
            id: 'project-2',
            title: 'Aegis AI - Mental Health Companion using Agentic AI',
            category: 'Healthcare',
            domain: 'Mental Health & Wellness',
            description: 'A compassionate AI-powered mental health companion designed to support students in managing stress and emotional well-being. Features multilingual support (Tamil and English) and integrates multiple communication channels including web chat, WhatsApp, and Telegram for comprehensive student support.',
            image: 'videos/2.mp4',
            video: 'videos/2.mp4',
            technologies: ['Phidata', 'Agno Framework', 'OpenAI', 'DeepSeek Models', 'Twilio', 'Telegram API', 'WhatsApp API'],
            frameworks: ['Phidata', 'Agno Agentic AI Framework'],
            keyFeatures: [
                'ChatGPT-like conversational interface',
                'Multilingual support (Tamil & English)',
                'WhatsApp integration for parent notifications',
                'Telegram agent for 24/7 student support',
                'Stress management and emotional support',
                'Privacy-focused mental health assistance'
            ],
            technicalDetails: {
                architecture: 'Agentic AI with Multi-Channel Integration',
                framework: 'Phidata (Agno)',
                llm: 'OpenAI + DeepSeek models',
                messaging: 'Twilio for WhatsApp & SMS',
                languages: 'English, Tamil',
                deployment: 'Multi-platform (Web, WhatsApp, Telegram)'
            },
            applications: [
                'Student mental health support',
                'Educational institution wellness programs',
                'Parent-student communication bridge',
                'Crisis intervention and support'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/aegis-ai',
            demoLink: '',
            tags: ['Healthcare', 'Mental Health', 'Agentic AI', 'Multilingual', 'Student Support', 'Phidata'],
            featured: true,
            projectNumber: 2,
            totalProjects: 5
        },
        {
            id: 'project-3',
            title: 'Project Pilot - Autonomous Application Builder',
            category: 'Application Development',
            domain: 'AI-Powered Development',
            description: 'An intelligent application building system powered by LangGraph featuring 5 specialized AI agents that collaborate to automate the entire software development lifecycle from planning to deployment and monitoring.',
            image: 'videos/3.mp4',
            video: 'videos/3.mp4',
            technologies: ['LangGraph', 'AI Agents', 'Python', 'LangChain', 'Autonomous Systems'],
            frameworks: ['LangGraph', 'LangChain'],
            keyFeatures: [
                'Planning Agent - Requirements analysis and architecture design',
                'Development Agent - Code generation and implementation',
                'Testing Agent - Automated testing and quality assurance',
                'Deployment Agent - CI/CD and production deployment',
                'Monitoring Agent - Performance tracking and optimization',
                'Multi-agent orchestration with LangGraph'
            ],
            technicalDetails: {
                architecture: 'Multi-Agent Orchestration System',
                agents: '5 specialized agents',
                framework: 'LangGraph for agent coordination',
                workflow: 'Planning → Development → Testing → Deployment → Monitoring',
                automation: 'End-to-end development automation',
                collaboration: 'Inter-agent communication and task delegation'
            },
            applications: [
                'Rapid application prototyping',
                'Automated software development',
                'DevOps workflow automation',
                'AI-assisted coding platforms'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/project-pilot',
            demoLink: '',
            tags: ['LangGraph', 'Multi-Agent System', 'Application Builder', 'DevOps', 'Automation', 'AI Development'],
            featured: true,
            projectNumber: 3,
            totalProjects: 5
        },
        {
            id: 'project-4',
            title: 'Network Security - End-to-End ML Project',
            category: 'Machine Learning / MLOps',
            domain: 'Cybersecurity',
            description: 'A comprehensive end-to-end machine learning project for network security deployed on AWS infrastructure. Built with modular programming principles, featuring complete MLOps pipeline with experiment tracking, containerization, and cloud deployment on AWS EC2 and ECR.',
            image: 'videos/4.mp4',
            video: 'videos/4.mp4',
            technologies: ['Python', 'Docker', 'DagsHub', 'MLflow', 'AWS EC2', 'AWS ECR', 'Machine Learning', 'CI/CD'],
            frameworks: ['MLflow', 'Docker', 'AWS'],
            keyFeatures: [
                'Fully modular programming architecture',
                'Complete MLOps pipeline implementation',
                'Experiment tracking with MLflow and DagsHub',
                'Containerized deployment with Docker',
                'AWS EC2 for scalable compute',
                'AWS ECR for container registry',
                'CI/CD automation for model deployment'
            ],
            technicalDetails: {
                architecture: 'Modular ML Pipeline',
                mlops: 'MLflow for experiment tracking',
                containerization: 'Docker for reproducibility',
                deployment: 'AWS EC2 + ECR',
                versionControl: 'DagsHub for ML versioning',
                cicd: 'Automated deployment pipeline',
                scalability: 'Cloud-native architecture'
            },
            applications: [
                'Network intrusion detection',
                'Cybersecurity threat analysis',
                'Anomaly detection in network traffic',
                'Enterprise security monitoring'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/network-security-ml',
            demoLink: '',
            tags: ['MLOps', 'Network Security', 'AWS', 'Docker', 'MLflow', 'End-to-End ML', 'Cloud Deployment'],
            featured: true,
            projectNumber: 4,
            totalProjects: 5
        },
        {
            id: 'project-5',
            title: 'Eye-Based Virtual Mouse using MediaPipe',
            category: 'Computer Vision',
            domain: 'Human-Computer Interaction',
            description: 'An innovative hands-free computer control system using eye tracking with MediaPipe. Control your cursor with left eye movements and perform clicks with right eye blinks, enabling accessible computing for users with limited mobility.',
            image: 'videos/5.mp4',
            video: 'videos/5.mp4',
            technologies: ['Python', 'MediaPipe', 'OpenCV', 'PyAutoGUI', 'Computer Vision', 'Eye Tracking'],
            frameworks: ['MediaPipe', 'OpenCV'],
            keyFeatures: [
                'Real-time eye tracking with MediaPipe',
                'Left eye controls cursor movement',
                'Right eye blink triggers mouse click',
                'Hands-free computer interaction',
                'Low-latency response system',
                'Accessibility-focused design',
                'PyAutoGUI integration for system control'
            ],
            technicalDetails: {
                architecture: 'Real-time Eye Tracking System',
                eyeTracking: 'MediaPipe Face Mesh',
                cursorControl: 'Left eye gaze mapping',
                clickDetection: 'Right eye blink recognition',
                automation: 'PyAutoGUI for mouse control',
                performance: 'Real-time processing with low latency',
                accessibility: 'Assistive technology implementation'
            },
            applications: [
                'Assistive technology for disabled users',
                'Hands-free computer control',
                'Accessibility solutions',
                'Gaming and entertainment',
                'Medical rehabilitation tools'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/eye-virtual-mouse',
            demoLink: '',
            tags: ['Computer Vision', 'Eye Tracking', 'MediaPipe', 'Accessibility', 'PyAutoGUI', 'Assistive Technology'],
            featured: true,
            projectNumber: 5,
            totalProjects: 5
        }
    ]
};

// Export for use in portfolio
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}