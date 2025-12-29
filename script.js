// Wait for database to load
function waitForDatabase() {
    if (typeof projectDatabase === 'undefined') {
        setTimeout(waitForDatabase, 50);
        return;
    }
    // Add a small delay to ensure DOM is fully ready after DB load
    setTimeout(initPortfolio, 100);
}

function initPortfolio() {
    updateCounts();
    renderProjects();
    setupEventListeners();
}

function updateCounts() {
    // For the new structure with 'all' array
    const total = projectDatabase.all ? projectDatabase.all.length : 0;

    const totalEl = document.getElementById('count-all');
    if (totalEl) totalEl.textContent = total;

    // Reset category counts to 0 since we're using a simple 'all' structure
    const categories = ['ml', 'dl', 'cv', 'nlp', 'genai', 'analytics'];
    categories.forEach(cat => {
        const el = document.getElementById(`count-${cat}`);
        if (el) el.textContent = 0;
    });
}

function renderProjects() {
    const grid = document.getElementById('projectsGrid');
    if (!grid) {
        console.error('projectsGrid element not found!');
        return;
    }

    const projects = projectDatabase.all || [];

    if (projects.length === 0) {
        grid.innerHTML = '<p style="text-align: center; color: #888;">No projects available</p>';
        return;
    }

    grid.innerHTML = projects.map(project => {
        return `
            <div class="project-card" onclick="showProjectPreview('${project.id}')">
                <div class="project-media">
                    <video class="project-video" muted loop playsinline onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0;">
                        <source src="${project.video}" type="video/mp4">
                    </video>
                    <div class="project-overlay">
                        <div class="project-category">${project.category}</div>
                    </div>
                </div>
                <div class="project-info">
                    <h3 class="project-title">${project.title}</h3>
                    <p class="project-description">${project.description.substring(0, 120)}...</p>
                    <div class="project-tech">
                        ${project.technologies.slice(0, 3).map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                        ${project.technologies.length > 3 ? `<span class="tech-tag">+${project.technologies.length - 3} more</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function setupEventListeners() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            currentCategory = e.currentTarget.dataset.category;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            renderProjects();
        });
    });

    window.addEventListener('click', (e) => {
        if (e.target === document.getElementById('projectPreviewModal')) {
            closeProjectPreview();
        }
    });
}

function showProjectPreview(projectId) {
    const projects = projectDatabase.all || [];
    const project = projects.find(p => p.id === projectId);

    if (!project) return;

    const modal = document.getElementById('projectPreviewModal');
    const title = document.getElementById('previewTitle');
    const body = document.getElementById('previewBody');

    title.textContent = project.title;

    body.innerHTML = `
        <div class="preview-media">
            <video controls autoplay loop class="preview-video" style="width: 100%; border-radius: 12px;">
                <source src="${project.video}" type="video/mp4">
                Your browser does not support the video tag.
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
                <h4><i class="fas fa-star"></i> Key Features</h4>
                <ul style="list-style: none; padding: 0;">
                    ${project.keyFeatures.map(feature => `<li style="padding: 8px 0; border-bottom: 1px solid #eee;"><i class="fas fa-check-circle" style="color: #00ffff; margin-right: 8px;"></i>${feature}</li>`).join('')}
                </ul>
            </div>
            ${project.githubLink ? `
            <div class="detail-section">
                <a href="${project.githubLink}" target="_blank" class="cta-primary" style="display: inline-block; margin-right: 10px;">
                    <i class="fab fa-github"></i> View on GitHub
                </a>
            </div>
            ` : ''}
        </div>
    `;

    modal.style.display = 'block';
}

function closeProjectPreview() {
    document.getElementById('projectPreviewModal').style.display = 'none';
}

function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
}

function downloadResume() {
    const link = document.createElement('a');
    link.href = 'Rajkumar V - AI Engineer Resume.pdf';
    link.download = 'Rajkumar_AI_Engineer_Resume.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

let currentCategory = 'all';

// Start when page loads
document.addEventListener('DOMContentLoaded', waitForDatabase);