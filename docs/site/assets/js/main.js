/**
 * Main JavaScript - Core Functionality
 * Handles count-up animations, smooth scrolling, and initialization
 */

document.addEventListener('DOMContentLoaded', () => {
  initCountUpAnimations();
  initSmoothScrolling();
  initNavigation();
});

/**
 * Count-up animation for statistics
 */
function initCountUpAnimations() {
  const countElements = document.querySelectorAll('.count-up');
  
  const observerOptions = {
    threshold: 0.5,
    rootMargin: '0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
        entry.target.classList.add('counted');
        animateCount(entry.target);
      }
    });
  }, observerOptions);

  countElements.forEach(el => observer.observe(el));
}

function animateCount(element) {
  const target = parseInt(element.getAttribute('data-target'));
  const duration = 2000; // 2 seconds
  const increment = target / (duration / 16); // 60fps
  let current = 0;

  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      current = target;
      clearInterval(timer);
    }
    element.textContent = Math.floor(current).toLocaleString();
  }, 16);
}

/**
 * Smooth scrolling for navigation links
 */
function initSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      
      if (target) {
        const offsetTop = target.offsetTop - 80; // Account for fixed navbar
        window.scrollTo({
          top: offsetTop,
          behavior: 'smooth'
        });
      }
    });
  });
}

/**
 * Navigation bar behavior
 */
function initNavigation() {
  const navbar = document.getElementById('navbar');
  const progressBar = document.getElementById('progress-bar');
  let lastScroll = 0;

  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    // Show/hide navbar on scroll
    if (currentScroll > 100) {
      navbar.classList.add('scrolled');
      
      if (currentScroll > lastScroll && currentScroll > 500) {
        navbar.classList.add('hidden');
      } else {
        navbar.classList.remove('hidden');
      }
    } else {
      navbar.classList.remove('scrolled');
      navbar.classList.remove('hidden');
    }

    // Update progress bar
    const windowHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrolled = (currentScroll / windowHeight) * 100;
    progressBar.style.width = scrolled + '%';

    lastScroll = currentScroll;
  });
}

/**
 * Lazy load iframes when they come into view
 */
function initLazyLoadIframes() {
  const iframes = document.querySelectorAll('iframe[data-src]');
  
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '50px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const iframe = entry.target;
        iframe.src = iframe.getAttribute('data-src');
        iframe.removeAttribute('data-src');
        observer.unobserve(iframe);
      }
    });
  }, observerOptions);

  iframes.forEach(iframe => observer.observe(iframe));
}

/**
 * Handle external links
 */
function initExternalLinks() {
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.hostname.includes(window.location.hostname)) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', () => {
  initLazyLoadIframes();
  initExternalLinks();
});