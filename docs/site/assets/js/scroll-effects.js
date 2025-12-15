/**
 * Scroll Effects - Intersection Observer animations
 * Handles fade-in, stagger effects, and scroll-triggered animations
 */

document.addEventListener('DOMContentLoaded', () => {
  initScrollAnimations();
  initActiveNavigation();
});

/**
 * Initialize scroll-based animations using Intersection Observer
 */
function initScrollAnimations() {
  const animatedElements = document.querySelectorAll(
    '.fade-in, .fade-in-up, .stagger-1, .stagger-2, .stagger-3, .stagger-4, .stagger-5'
  );

  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        // Optionally unobserve after animation
        // observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  animatedElements.forEach(el => {
    observer.observe(el);
  });
}

/**
 * Highlight active navigation section
 */
function initActiveNavigation() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a');

  const observerOptions = {
    threshold: 0.3,
    rootMargin: '-80px 0px -70% 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const currentId = entry.target.getAttribute('id');
        
        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${currentId}`) {
            link.classList.add('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));
}

/**
 * Parallax effect for hero section
 */
function initParallax() {
  const hero = document.getElementById('hero');
  
  if (!hero) return;

  window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const rate = scrolled * 0.5;
    
    hero.style.transform = `translate3d(0, ${rate}px, 0)`;
  });
}

/**
 * Add reveal animation class when element is visible
 */
function revealOnScroll(selector, animationClass = 'revealed') {
  const elements = document.querySelectorAll(selector);
  
  const observerOptions = {
    threshold: 0.15
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add(animationClass);
      }
    });
  }, observerOptions);

  elements.forEach(el => observer.observe(el));
}

// Initialize parallax if desired (optional)
// initParallax();