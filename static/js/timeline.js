document.addEventListener('DOMContentLoaded', function() {
    const timelineItems = document.querySelectorAll('.timeline-item');
    const contentView = document.querySelector('.content-view');
    const themeToggle = document.getElementById('theme-toggle');
    const welcomeMessage = document.querySelector('.welcome-message');
    let activeItem = null;

    // Theme toggle functionality
    function setTheme(isDark) {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        themeToggle.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    }

    // Initialize theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme === 'dark');

    themeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'light';
        setTheme(isDark);
    });

    // Function to load content
    async function loadContent(articleId) {
        try {
            if (welcomeMessage) {
                welcomeMessage.style.display = 'none';
            }
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading';
            loadingDiv.textContent = 'Loading content...';
            contentView.appendChild(loadingDiv);

            const response = await fetch(`/content/${articleId}`);
            const data = await response.json();

            if (data.success) {
                // Remove loading message
                if (loadingDiv) {
                    loadingDiv.remove();
                }

                // Create an iframe to display the content
                const iframe = document.createElement('iframe');
                iframe.srcdoc = data.content;

                // Add transition effect
                iframe.style.opacity = '0';
                iframe.style.transition = 'opacity 0.3s ease';

                contentView.innerHTML = '';
                contentView.appendChild(iframe);

                // Handle link clicks within the iframe
                iframe.onload = function() {
                    const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                    iframeDoc.querySelectorAll('a').forEach(link => {
                        link.addEventListener('click', function(e) {
                            e.preventDefault();
                            const href = this.getAttribute('href');
                            if (href && !href.startsWith('#')) {
                                window.open(href, '_blank');
                            } else if (href && href.startsWith('#')) {
                                // Handle internal anchor links
                                const targetId = href.substring(1);
                                const targetElement = iframeDoc.getElementById(targetId);
                                if (targetElement) {
                                    targetElement.scrollIntoView({ behavior: 'smooth' });
                                }
                            }
                        });
                    });
                };

                // Trigger reflow and add opacity
                setTimeout(() => {
                    iframe.style.opacity = '1';
                }, 50);
            } else {
                contentView.innerHTML = '<div class="error">Failed to load content</div>';
            }
        } catch (error) {
            console.error('Error loading content:', error);
            contentView.innerHTML = '<div class="error">Error loading content</div>';
        }
    }

    // Add click handlers for timeline items
    timelineItems.forEach(item => {
        item.addEventListener('click', function() {
            const articleId = this.dataset.articleId;
            if (articleId) {
                if (activeItem) {
                    activeItem.classList.remove('active');
                }
                this.classList.add('active');
                activeItem = this;
                loadContent(articleId);
            }
        });
    });

    // Load the first item by default
    if (timelineItems.length > 0) {
        timelineItems[0].click();
    }

    // Handle window resize for responsive iframe
    window.addEventListener('resize', function() {
        const iframe = contentView.querySelector('iframe');
        if (iframe) {
            iframe.style.height = `${window.innerHeight - 100}px`;
        }
    });

    // Add hover effects
    timelineItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            if (this !== activeItem) {
                this.style.transform = 'translateX(5px)';
            }
        });

        item.addEventListener('mouseleave', function() {
            if (this !== activeItem) {
                this.style.transform = 'translateX(0)';
            }
        });
    });

    // Handle mobile view
    function handleMobileView() {
        const timelineContainer = document.querySelector('.timeline-container');
        const contentContainer = document.querySelector('.content-container');

        if (window.innerWidth <= 768) {
            timelineItems.forEach(item => {
                item.addEventListener('click', () => {
                    timelineContainer.classList.add('collapsed');
                    contentContainer.classList.add('expanded');
                });
            });
        }
    }

    handleMobileView();
    window.addEventListener('resize', handleMobileView);
});
