document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const articlesContainer = document.getElementById('articles-container');
    const exportForm = document.getElementById('export-form');
    const tagModal = new bootstrap.Modal(document.getElementById('tagModal'));
    let currentArticleId = null;

    // Search functionality
    searchInput.addEventListener('input', debounce(function() {
        const query = this.value.trim();
        if (query.length > 2) {
            fetch(`/search?q=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    // Update articles display with search results
                    updateArticlesDisplay(data.results);
                });
        }
    }, 300));

    // Export functionality
    exportForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        fetch('/export', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showAlert('success', data.message);
            } else {
                showAlert('danger', data.message);
            }
        });
    });

    // Add tag button click
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('add-tag-btn')) {
            currentArticleId = e.target.dataset.articleId;
            tagModal.show();
        }
    });

    // Save tags
    document.getElementById('save-tags-btn').addEventListener('click', function() {
        const tags = document.getElementById('tag-input').value;
        
        fetch('/add_tag', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `article_id=${currentArticleId}&tags=${encodeURIComponent(tags)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                tagModal.hide();
                location.reload();
            } else {
                showAlert('danger', data.message);
            }
        });
    });

    // Remove favorite
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('remove-favorite-btn')) {
            const articleId = e.target.dataset.articleId;
            if (confirm('Are you sure you want to remove this article from favorites?')) {
                window.location.href = `/favorite/${articleId}`;
            }
        }
    });

    // Utility functions
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    function showAlert(type, message) {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.querySelector('.container').insertBefore(alert, document.querySelector('.row'));
        setTimeout(() => alert.remove(), 5000);
    }

    function updateArticlesDisplay(articles) {
        articlesContainer.innerHTML = articles.map(article => `
            <div class="card mb-3 article-card">
                <div class="card-body">
                    <h5 class="card-title">${article.title}</h5>
                    <p class="card-text">${article.summary}</p>
                    <div class="article-tags mb-2">
                        ${article.tags.map(tag => `
                            <span class="badge bg-secondary me-1">${tag}</span>
                        `).join('')}
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-outline-primary add-tag-btn" 
                                data-article-id="${article.id}">
                            Add Tag
                        </button>
                        <a href="${article.url}" target="_blank" 
                           class="btn btn-sm btn-outline-secondary">
                            Read Original
                        </a>
                        <button class="btn btn-sm btn-outline-danger remove-favorite-btn"
                                data-article-id="${article.id}">
                            Remove from Favorites
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    }
});
