// AI Portfolio Manager - Optimized Version
class AIPortfolio {
    constructor() {
        this.currentCategory = 'all';
        this.projects = [];
        this.init();
    }

    init() {
        if (typeof projectDatabase === 'undefined') {
            setTimeout(() => this.init(), 50);
            return;
        }
        this.projects = projectDatabase.all || [];
        this.updateCounts();
        this.renderProjects();
        this.setupEventListeners();
        this.initSkillBars();
        this.createParticles();
    }

    updateCounts() {
        // Counts removed - no longer needed
    }

    renderProjects() {
        const grid = document.getElementById('projectsGrid');
        const emptyState = document.getElementById('emptyState');
        
        if (!grid) return;

        let filteredProjects = this.projects;

        if (filteredProjects.length === 0) {
            grid.style.display = 'none';
            if (emptyState) emptyState.style.display = 'block';
            return;
        }

        grid.style.display = 'grid';
        if (emptyState) emptyState.style.display = 'none';

        grid.innerHTML = filteredProjects.map(project => `
            <div class="project-card" data-category="${project.category?.toLowerCase()}">
                <div class="project-media">
                    <video class="project-video" muted loop playsinline 
                           onmouseover="this.play()" 
                           onmouseout="this.pause();this.currentTime=0;">
                        <source src="${project.video}" type="video/mp4">
                    </video>
                </div>
                <div class="project-content">
                    <div class="project-header">
                        <h3 class="project-title">${project.title}</h3>
                        <div class="project-tags">
                            <span class="project-tag">${project.category}</span>
                            ${project.domain ? `<span class="project-tag">${project.domain}</span>` : ''}
                        </div>
                    </div>
                    <p class="project-description">${project.description}</p>
                    <div class="project-tech">
                        <h4><i class="fas fa-code"></i> Tech Stack</h4>
                        <p class="project-tech-list">${project.technologies.slice(0, 4).join(', ')}${project.technologies.length > 4 ? '...' : ''}</p>
                    </div>
                    <div class="project-actions">
                        <button class="project-btn primary" onclick="portfolio.showPreview('${project.id}')">
                            <i class="fas fa-eye"></i> View Details
                        </button>
                        ${project.githubLink ? `
                            <a href="${project.githubLink}" target="_blank" class="project-btn secondary">
                                <i class="fab fa-github"></i> GitHub
                            </a>
                        ` : ''}
                    </div>
                </div>
            </div>
        `).join('');
    }

    showPreview(projectId) {
        const project = this.projects.find(p => p.id === projectId);
        if (!project) return;

        const modal = document.getElementById('projectPreviewModal');
        const title = document.getElementById('previewTitle');
        const body = document.getElementById('previewBody');

        title.textContent = project.title;
        body.innerHTML = `
            <div class="preview-media">
                <video controls autoplay loop style="width: 100%; border-radius: 12px;">
                    <source src="${project.video}" type="video/mp4">
                </video>
            </div>
            <div class="preview-details">
                <div class="detail-section">
                    <h4><i class="fas fa-info-circle"></i> Description</h4>
                    <p>${project.description}</p>
                </div>
                <div class="detail-section">
                    <h4><i class="fas fa-layer-group"></i> Domain</h4>
                    <p>${project.domain}</p>
                </div>
                <div class="detail-section">
                    <h4><i class="fas fa-code"></i> Technologies</h4>
                    <div class="tech-tags">
                        ${project.technologies.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                    </div>
                </div>
                <div class="detail-section">
                    <h4><i class="fas fa-tools"></i> Frameworks</h4>
                    <div class="tech-tags">
                        ${project.frameworks.map(fw => `<span class="tech-tag">${fw}</span>`).join('')}
                    </div>
                </div>
                <div class="detail-section">
                    <h4><i class="fas fa-star"></i> Key Features</h4>
                    <ul style="list-style: none; padding: 0;">
                        ${project.keyFeatures.map(feature => `
                            <li style="padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <i class="fas fa-check-circle" style="color: #00ffff; margin-right: 8px;"></i>${feature}
                            </li>
                        `).join('')}
                    </ul>
                </div>
                <div class="detail-section">
                    <h4><i class="fas fa-rocket"></i> Applications</h4>
                    <ul style="list-style: none; padding: 0;">
                        ${project.applications.map(app => `
                            <li style="padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <i class="fas fa-arrow-right" style="color: #00ffff; margin-right: 8px;"></i>${app}
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ${project.githubLink ? `
                    <div class="detail-section">
                        <a href="${project.githubLink}" target="_blank" class="cta-primary" style="display: inline-flex; align-items: center; gap: 0.5rem; text-decoration: none;">
                            <i class="fab fa-github"></i> View on GitHub
                        </a>
                    </div>
                ` : ''}
            </div>
        `;

        modal.style.display = 'block';
    }

    closePreview() {
        document.getElementById('projectPreviewModal').style.display = 'none';
    }

    setupEventListeners() {
        // Modal close
        window.addEventListener('click', (e) => {
            if (e.target === document.getElementById('projectPreviewModal')) {
                this.closePreview();
            }
        });
    }

    initSkillBars() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const bar = entry.target;
                    const width = bar.dataset.width;
                    setTimeout(() => {
                        bar.style.width = width + '%';
                    }, 100);
                    observer.unobserve(bar);
                }
            });
        }, { threshold: 0.5 });

        document.querySelectorAll('.skill-progress').forEach(bar => {
            observer.observe(bar);
        });
    }

    createParticles() {
        const container = document.getElementById('particles');
        if (!container) return;

        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.width = Math.random() * 5 + 2 + 'px';
            particle.style.height = particle.style.width;
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 6 + 's';
            particle.style.animationDuration = Math.random() * 4 + 4 + 's';
            container.appendChild(particle);
        }
    }
}

// Global functions
function scrollToSection(sectionId) {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
}

function downloadResume() {
    const link = document.createElement('a');
    link.href = 'Rajkumar V - AI Engineer Resume.pdf';
    link.download = 'Rajkumar_AI_Engineer_Resume.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function closeProjectPreview() {
    portfolio.closePreview();
}

// Initialize portfolio
let portfolio;
document.addEventListener('DOMContentLoaded', () => {
    portfolio = new AIPortfolio();
});