/**
 * Particle Background Animation
 * Creates animated particle network in hero section
 */

(function() {
  const canvas = document.getElementById('particles-background');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let particles = [];
  let animationFrameId;

  // Configuration
  const config = {
    particleCount: 80,
    particleSize: 2,
    particleSpeed: 0.3,
    connectionDistance: 150,
    particleColor: 'rgba(255, 69, 0, 0.6)', // Reddit orange
    lineColor: 'rgba(255, 69, 0, 0.15)',
    backgroundColor: '#030303'
  };

  // Resize canvas to window size
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  // Particle class
  class Particle {
    constructor() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.vx = (Math.random() - 0.5) * config.particleSpeed;
      this.vy = (Math.random() - 0.5) * config.particleSpeed;
      this.radius = config.particleSize;
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;

      // Bounce off edges
      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;

      // Keep within bounds
      this.x = Math.max(0, Math.min(canvas.width, this.x));
      this.y = Math.max(0, Math.min(canvas.height, this.y));
    }

    draw() {
      ctx.fillStyle = config.particleColor;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Initialize particles
  function initParticles() {
    particles = [];
    for (let i = 0; i < config.particleCount; i++) {
      particles.push(new Particle());
    }
  }

  // Draw connections between nearby particles
  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < config.connectionDistance) {
          const opacity = 1 - (distance / config.connectionDistance);
          ctx.strokeStyle = `rgba(255, 69, 0, ${opacity * 0.15})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  // Animation loop
  function animate() {
    ctx.fillStyle = config.backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    drawConnections();

    particles.forEach(particle => {
      particle.update();
      particle.draw();
    });

    animationFrameId = requestAnimationFrame(animate);
  }

  // Mouse interaction (optional)
  function initMouseInteraction() {
    let mouse = { x: null, y: null, radius: 150 };

    canvas.addEventListener('mousemove', (e) => {
      mouse.x = e.x;
      mouse.y = e.y;
    });

    canvas.addEventListener('mouseleave', () => {
      mouse.x = null;
      mouse.y = null;
    });

    // Modify particle behavior based on mouse proximity
    setInterval(() => {
      if (mouse.x !== null && mouse.y !== null) {
        particles.forEach(particle => {
          const dx = mouse.x - particle.x;
          const dy = mouse.y - particle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < mouse.radius) {
            const force = (mouse.radius - distance) / mouse.radius;
            particle.vx -= (dx / distance) * force * 0.1;
            particle.vy -= (dy / distance) * force * 0.1;
          }
        });
      }
    }, 50);
  }

  // Initialize and start
  function init() {
    resizeCanvas();
    initParticles();
    animate();
    initMouseInteraction();

    window.addEventListener('resize', () => {
      resizeCanvas();
      initParticles();
    });
  }

  // Pause animation when not visible (performance)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(animationFrameId);
    } else {
      animate();
    }
  });

  init();
})();