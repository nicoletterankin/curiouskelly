// Parallax Effect for VisionOS-style Depth Layers

class ParallaxController {
    constructor() {
        this.mouseX = 0;
        this.mouseY = 0;
        this.parallaxElements = [];
        this.isActive = false;
        this.init();
    }

    init() {
        // Find all parallax elements
        this.parallaxElements = document.querySelectorAll('.parallax-layer');
        
        // Add mouse move listener
        document.addEventListener('mousemove', (e) => {
            this.handleMouseMove(e);
        });
        
        // Add touch move listener for mobile
        document.addEventListener('touchmove', (e) => {
            if (e.touches.length > 0) {
                this.handleMouseMove({
                    clientX: e.touches[0].clientX,
                    clientY: e.touches[0].clientY
                });
            }
        });
        
        this.isActive = true;
    }

    handleMouseMove(e) {
        this.mouseX = e.clientX;
        this.mouseY = e.clientY;
        
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        
        const deltaX = (this.mouseX - centerX) / centerX;
        const deltaY = (this.mouseY - centerY) / centerY;
        
        this.parallaxElements.forEach((element, index) => {
            const depth = parseFloat(element.dataset.depth || '0.1');
            const intensity = parseFloat(element.dataset.intensity || '10');
            
            const translateX = deltaX * intensity * depth;
            const translateY = deltaY * intensity * depth;
            
            element.style.transform = `translate3d(${translateX}px, ${translateY}px, 0)`;
        });
    }

    addElement(element, depth = 0.1, intensity = 10) {
        element.classList.add('parallax-layer');
        element.dataset.depth = depth;
        element.dataset.intensity = intensity;
        this.parallaxElements = document.querySelectorAll('.parallax-layer');
    }

    removeElement(element) {
        element.classList.remove('parallax-layer');
        this.parallaxElements = document.querySelectorAll('.parallax-layer');
    }
}

// Initialize parallax controller
let parallaxController;
if (typeof window !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        parallaxController = new ParallaxController();
    });
}








