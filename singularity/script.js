/**
 * SINGULARITY — GPU Fluid Simulation + Particle Galaxy
 * WebGL Navier-Stokes fluid with post-processing bloom
 */

// ─── WebGL Fluid Simulation ───────────────────────────────────────
const fluidCanvas = document.getElementById('fluid-canvas');
const gl = fluidCanvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
if (!gl) alert('WebGL2 required');

const partCanvas = document.getElementById('particle-canvas');
const pCtx = partCanvas.getContext('2d');

let mouse = { x: 0, y: 0, px: 0, py: 0, down: false, moved: false };
let simW, simH, dispW, dispH;
const SIM_SCALE = 0.25; // fluid sim at quarter res for perf

// ─── Shader Sources ───────────────────────────────────────────────
const VERT = `#version 300 es
in vec2 a_pos;
out vec2 vUv;
void main(){
    vUv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0, 1);
}`;

const ADVECT = `#version 300 es
precision highp float;
uniform sampler2D u_vel;
uniform sampler2D u_source;
uniform vec2 u_texel;
uniform float u_dt;
uniform float u_dissipation;
in vec2 vUv;
out vec4 fragColor;
void main(){
    vec2 vel = texture(u_vel, vUv).xy;
    vec2 coord = vUv - vel * u_texel * u_dt;
    fragColor = u_dissipation * texture(u_source, coord);
}`;

const DIVERGENCE = `#version 300 es
precision highp float;
uniform sampler2D u_vel;
uniform vec2 u_texel;
in vec2 vUv;
out vec4 fragColor;
void main(){
    float L = texture(u_vel, vUv - vec2(u_texel.x,0)).x;
    float R = texture(u_vel, vUv + vec2(u_texel.x,0)).x;
    float B = texture(u_vel, vUv - vec2(0,u_texel.y)).y;
    float T = texture(u_vel, vUv + vec2(0,u_texel.y)).y;
    fragColor = vec4(0.5*(R-L+T-B), 0, 0, 1);
}`;

const PRESSURE = `#version 300 es
precision highp float;
uniform sampler2D u_pressure;
uniform sampler2D u_div;
uniform vec2 u_texel;
in vec2 vUv;
out vec4 fragColor;
void main(){
    float L = texture(u_pressure, vUv - vec2(u_texel.x,0)).x;
    float R = texture(u_pressure, vUv + vec2(u_texel.x,0)).x;
    float B = texture(u_pressure, vUv - vec2(0,u_texel.y)).x;
    float T = texture(u_pressure, vUv + vec2(0,u_texel.y)).x;
    float d = texture(u_div, vUv).x;
    fragColor = vec4((L+R+B+T-d)*0.25, 0, 0, 1);
}`;

const GRADIENT_SUB = `#version 300 es
precision highp float;
uniform sampler2D u_pressure;
uniform sampler2D u_vel;
uniform vec2 u_texel;
in vec2 vUv;
out vec4 fragColor;
void main(){
    float L = texture(u_pressure, vUv - vec2(u_texel.x,0)).x;
    float R = texture(u_pressure, vUv + vec2(u_texel.x,0)).x;
    float B = texture(u_pressure, vUv - vec2(0,u_texel.y)).x;
    float T = texture(u_pressure, vUv + vec2(0,u_texel.y)).x;
    vec2 vel = texture(u_vel, vUv).xy - vec2(R-L, T-B)*0.5;
    fragColor = vec4(vel, 0, 1);
}`;

const SPLAT = `#version 300 es
precision highp float;
uniform sampler2D u_target;
uniform vec2 u_point;
uniform vec3 u_color;
uniform float u_radius;
uniform float u_aspect;
in vec2 vUv;
out vec4 fragColor;
void main(){
    vec2 p = vUv - u_point;
    p.x *= u_aspect;
    float d = dot(p,p);
    vec3 splat = u_color * exp(-d / u_radius);
    vec3 base = texture(u_target, vUv).xyz;
    fragColor = vec4(base + splat, 1);
}`;

const DISPLAY = `#version 300 es
precision highp float;
uniform sampler2D u_texture;
uniform sampler2D u_bloom;
in vec2 vUv;
out vec4 fragColor;

vec3 aces(vec3 x) {
    float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main(){
    vec3 col = texture(u_texture, vUv).rgb;
    vec3 bloom = texture(u_bloom, vUv).rgb;
    col += bloom * 0.6;
    col = aces(col);
    col = pow(col, vec3(1.0/2.2));
    
    // Chromatic aberration
    vec2 dir = vUv - 0.5;
    float d = length(dir) * 0.006;
    float r = texture(u_texture, vUv + dir*d).r;
    float b2 = texture(u_texture, vUv - dir*d).b;
    col.r = mix(col.r, r, 0.5);
    col.b = mix(col.b, b2, 0.5);
    
    fragColor = vec4(col, 1);
}`;

const BLUR = `#version 300 es
precision highp float;
uniform sampler2D u_texture;
uniform vec2 u_dir;
uniform vec2 u_texel;
in vec2 vUv;
out vec4 fragColor;
void main(){
    vec3 sum = vec3(0);
    sum += texture(u_texture, vUv - 4.0*u_dir*u_texel).rgb * 0.0162;
    sum += texture(u_texture, vUv - 3.0*u_dir*u_texel).rgb * 0.0540;
    sum += texture(u_texture, vUv - 2.0*u_dir*u_texel).rgb * 0.1216;
    sum += texture(u_texture, vUv - 1.0*u_dir*u_texel).rgb * 0.1945;
    sum += texture(u_texture, vUv).rgb * 0.2270;
    sum += texture(u_texture, vUv + 1.0*u_dir*u_texel).rgb * 0.1945;
    sum += texture(u_texture, vUv + 2.0*u_dir*u_texel).rgb * 0.1216;
    sum += texture(u_texture, vUv + 3.0*u_dir*u_texel).rgb * 0.0540;
    sum += texture(u_texture, vUv + 4.0*u_dir*u_texel).rgb * 0.0162;
    fragColor = vec4(sum, 1);
}`;

// ─── GL Helpers ───────────────────────────────────────────────────
function compileShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(s));
    return s;
}

function createProgram(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compileShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compileShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    const uniforms = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
        const info = gl.getActiveUniform(p, i);
        uniforms[info.name] = gl.getUniformLocation(p, info.name);
    }
    return { program: p, uniforms };
}

function createFBO(w, h, internalFormat, format, type, filter) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { texture: tex, fbo, w, h };
}

function createDoubleFBO(w, h) {
    const ext = gl.getExtension('EXT_color_buffer_float');
    const fmt = gl.RGBA16F, f = gl.RGBA, t = gl.HALF_FLOAT, fil = gl.LINEAR;
    let a = createFBO(w, h, fmt, f, t, fil);
    let b = createFBO(w, h, fmt, f, t, fil);
    return {
        get read() { return a; },
        get write() { return b; },
        swap() { [a, b] = [b, a]; }
    };
}

// fullscreen quad
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

function blit(target) {
    if (target) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        gl.viewport(0, 0, target.w, target.h);
    } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// ─── Init Programs & Buffers ──────────────────────────────────────
const advectProg = createProgram(VERT, ADVECT);
const divProg = createProgram(VERT, DIVERGENCE);
const presProg = createProgram(VERT, PRESSURE);
const gradProg = createProgram(VERT, GRADIENT_SUB);
const splatProg = createProgram(VERT, SPLAT);
const displayProg = createProgram(VERT, DISPLAY);
const blurProg = createProgram(VERT, BLUR);

let velocity, dye, divergence, pressure, bloomFBO, bloomTemp;

function resize() {
    dispW = window.innerWidth;
    dispH = window.innerHeight;
    fluidCanvas.width = dispW;
    fluidCanvas.height = dispH;
    partCanvas.width = dispW * devicePixelRatio;
    partCanvas.height = dispH * devicePixelRatio;
    partCanvas.style.width = dispW + 'px';
    partCanvas.style.height = dispH + 'px';
    pCtx.scale(devicePixelRatio, devicePixelRatio);

    simW = Math.floor(dispW * SIM_SCALE);
    simH = Math.floor(dispH * SIM_SCALE);

    velocity = createDoubleFBO(simW, simH);
    dye = createDoubleFBO(simW, simH);
    divergence = createFBO(simW, simH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    pressure = createDoubleFBO(simW, simH);
    bloomFBO = createFBO(simW, simH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
    bloomTemp = createFBO(simW, simH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
}

function useProg(p) {
    gl.useProgram(p.program);
    const loc = gl.getAttribLocation(p.program, 'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    return p.uniforms;
}

function splat(x, y, dx, dy, color) {
    let u = useProg(splatProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocity.read.texture);
    gl.uniform1i(u.u_target, 0);
    gl.uniform2f(u.u_point, x / dispW, 1.0 - y / dispH);
    gl.uniform3f(u.u_color, dx * 10, -dy * 10, 0);
    gl.uniform1f(u.u_radius, 0.0005);
    gl.uniform1f(u.u_aspect, dispW / dispH);
    blit(velocity.write);
    velocity.swap();

    u = useProg(splatProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, dye.read.texture);
    gl.uniform1i(u.u_target, 0);
    gl.uniform2f(u.u_point, x / dispW, 1.0 - y / dispH);
    gl.uniform3f(u.u_color, color[0], color[1], color[2]);
    gl.uniform1f(u.u_radius, 0.0008);
    gl.uniform1f(u.u_aspect, dispW / dispH);
    blit(dye.write);
    dye.swap();
}

function step(dt) {
    // Advect velocity
    let u = useProg(advectProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocity.read.texture);
    gl.uniform1i(u.u_vel, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, velocity.read.texture);
    gl.uniform1i(u.u_source, 1);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    gl.uniform1f(u.u_dt, dt);
    gl.uniform1f(u.u_dissipation, 0.99);
    blit(velocity.write);
    velocity.swap();

    // Advect dye
    u = useProg(advectProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocity.read.texture);
    gl.uniform1i(u.u_vel, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, dye.read.texture);
    gl.uniform1i(u.u_source, 1);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    gl.uniform1f(u.u_dt, dt);
    gl.uniform1f(u.u_dissipation, 0.97);
    blit(dye.write);
    dye.swap();

    // Divergence
    u = useProg(divProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocity.read.texture);
    gl.uniform1i(u.u_vel, 0);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    blit(divergence);

    // Pressure solve (Jacobi iteration)
    u = useProg(presProg);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    for (let i = 0; i < 20; i++) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, pressure.read.texture);
        gl.uniform1i(u.u_pressure, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, divergence.texture);
        gl.uniform1i(u.u_div, 1);
        blit(pressure.write);
        pressure.swap();
    }

    // Gradient subtraction
    u = useProg(gradProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pressure.read.texture);
    gl.uniform1i(u.u_pressure, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, velocity.read.texture);
    gl.uniform1i(u.u_vel, 1);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    blit(velocity.write);
    velocity.swap();
}

function bloom() {
    // Horizontal blur
    let u = useProg(blurProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, dye.read.texture);
    gl.uniform1i(u.u_texture, 0);
    gl.uniform2f(u.u_dir, 1, 0);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    blit(bloomTemp);
    // Vertical blur
    u = useProg(blurProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, bloomTemp.texture);
    gl.uniform1i(u.u_texture, 0);
    gl.uniform2f(u.u_dir, 0, 1);
    gl.uniform2f(u.u_texel, 1/simW, 1/simH);
    blit(bloomFBO);
}

function display() {
    bloom();
    let u = useProg(displayProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, dye.read.texture);
    gl.uniform1i(u.u_texture, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomFBO.texture);
    gl.uniform1i(u.u_bloom, 1);
    blit(null);
}

// ─── Particle System (Canvas 2D overlay) ──────────────────────────
const PARTICLE_COUNT = 3000;
const particles = [];

class Particle {
    constructor() { this.reset(true); }
    reset(initial) {
        this.x = initial ? Math.random() * dispW : mouse.x + (Math.random()-0.5)*100;
        this.y = initial ? Math.random() * dispH : mouse.y + (Math.random()-0.5)*100;
        this.vx = (Math.random()-0.5) * 2;
        this.vy = (Math.random()-0.5) * 2;
        this.life = 1;
        this.decay = 0.001 + Math.random()*0.003;
        this.size = 0.5 + Math.random()*1.5;
        this.hue = Math.random()*60 + 180; // cyan-blue range
    }
    update() {
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const dist = Math.sqrt(dx*dx + dy*dy) + 1;
        const force = mouse.down ? 800 : 120;
        const pull = force / (dist * dist) * (mouse.down ? 1 : -0.3);
        this.vx += dx / dist * pull;
        this.vy += dy / dist * pull;
        // Orbital tangent when close
        if (dist < 300) {
            this.vx += -dy / dist * 0.5;
            this.vy += dx / dist * 0.5;
        }
        this.vx *= 0.96;
        this.vy *= 0.96;
        this.x += this.vx;
        this.y += this.vy;
        this.life -= this.decay;
        if (this.life <= 0 || this.x < -50 || this.x > dispW+50 || this.y < -50 || this.y > dispH+50) {
            this.reset(false);
        }
    }
    draw(ctx) {
        const a = this.life * 0.6;
        ctx.fillStyle = `hsla(${this.hue}, 100%, 70%, ${a})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * this.life, 0, Math.PI*2);
        ctx.fill();
    }
}

// ─── Explosion Effect ─────────────────────────────────────────────
let explosions = [];
class Explosion {
    constructor(x, y) {
        this.x = x; this.y = y;
        this.radius = 0; this.maxR = 200 + Math.random()*200;
        this.life = 1; this.hue = Math.random()*360;
    }
    update() {
        this.radius += (this.maxR - this.radius) * 0.08;
        this.life *= 0.96;
    }
    draw(ctx) {
        ctx.strokeStyle = `hsla(${this.hue}, 100%, 60%, ${this.life*0.4})`;
        ctx.lineWidth = 2 * this.life;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI*2);
        ctx.stroke();
        ctx.strokeStyle = `hsla(${this.hue+30}, 100%, 80%, ${this.life*0.2})`;
        ctx.lineWidth = 8 * this.life;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius*0.6, 0, Math.PI*2);
        ctx.stroke();
    }
}

// ─── Events ───────────────────────────────────────────────────────
window.addEventListener('mousemove', e => {
    mouse.px = mouse.x; mouse.py = mouse.y;
    mouse.x = e.clientX; mouse.y = e.clientY;
    mouse.moved = true;
});
window.addEventListener('mousedown', () => { mouse.down = true; });
window.addEventListener('mouseup', () => { mouse.down = false; });
window.addEventListener('click', e => {
    explosions.push(new Explosion(e.clientX, e.clientY));
    // Massive fluid splat on click
    for (let i = 0; i < 8; i++) {
        const a = Math.PI*2 * i/8;
        const h = (Date.now()*0.05 + i*45) % 360;
        const r = Math.cos(h*Math.PI/180)*0.5+0.5;
        const g = Math.cos((h+120)*Math.PI/180)*0.5+0.5;
        const b = Math.cos((h+240)*Math.PI/180)*0.5+0.5;
        splat(e.clientX, e.clientY, Math.cos(a)*60, Math.sin(a)*60, [r*3,g*3,b*3]);
    }
});
window.addEventListener('touchmove', e => {
    const t = e.touches[0];
    mouse.px = mouse.x; mouse.py = mouse.y;
    mouse.x = t.clientX; mouse.y = t.clientY;
    mouse.moved = true;
}, {passive:true});
window.addEventListener('touchstart', e => {
    const t = e.touches[0];
    mouse.x = t.clientX; mouse.y = t.clientY;
    mouse.px = mouse.x; mouse.py = mouse.y;
    mouse.down = true;
});
window.addEventListener('touchend', () => { mouse.down = false; });
window.addEventListener('resize', resize);

// ─── Main Loop ────────────────────────────────────────────────────
let lastTime = 0;
let hueOffset = 0;

function animate(time) {
    requestAnimationFrame(animate);
    const dt = Math.min((time - lastTime) * 0.001, 0.033);
    lastTime = time;
    hueOffset = (time * 0.02) % 360;

    // Continuous cursor splat
    if (mouse.moved) {
        const dx = mouse.x - mouse.px;
        const dy = mouse.y - mouse.py;
        const h = hueOffset;
        const r = Math.cos(h*Math.PI/180)*0.5+0.5;
        const g = Math.cos((h+120)*Math.PI/180)*0.5+0.5;
        const b = Math.cos((h+240)*Math.PI/180)*0.5+0.5;
        const intensity = mouse.down ? 4 : 1.5;
        splat(mouse.x, mouse.y, dx, dy, [r*intensity, g*intensity, b*intensity]);
        mouse.moved = false;
    }

    // Auto ambient splats
    if (Math.random() < 0.03) {
        const ax = Math.random() * dispW;
        const ay = Math.random() * dispH;
        const h2 = (hueOffset + 180) % 360;
        splat(ax, ay, (Math.random()-0.5)*30, (Math.random()-0.5)*30, 
            [Math.cos(h2*Math.PI/180)*0.3+0.3, Math.cos((h2+120)*Math.PI/180)*0.3+0.3, Math.cos((h2+240)*Math.PI/180)*0.3+0.3]);
    }

    step(dt * 8);
    display();

    // Particle overlay
    pCtx.clearRect(0, 0, dispW, dispH);
    pCtx.globalCompositeOperation = 'lighter';
    for (const p of particles) { p.update(); p.draw(pCtx); }
    
    // Explosions
    explosions = explosions.filter(e => e.life > 0.01);
    for (const e of explosions) { e.update(); e.draw(pCtx); }

    // Cursor glow
    const cg = pCtx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, mouse.down ? 80 : 40);
    cg.addColorStop(0, `hsla(${hueOffset}, 100%, 90%, ${mouse.down ? 0.6 : 0.3})`);
    cg.addColorStop(1, 'transparent');
    pCtx.globalCompositeOperation = 'lighter';
    pCtx.fillStyle = cg;
    pCtx.fillRect(mouse.x-100, mouse.y-100, 200, 200);
}

// ─── Boot ─────────────────────────────────────────────────────────
resize();
mouse.x = dispW/2; mouse.y = dispH/2; mouse.px = mouse.x; mouse.py = mouse.y;
for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(new Particle());
// Initial burst
for (let i = 0; i < 20; i++) {
    const a = Math.PI*2*i/20;
    splat(dispW/2, dispH/2, Math.cos(a)*40, Math.sin(a)*40, [0.3,0.5,1]);
}
requestAnimationFrame(animate);
