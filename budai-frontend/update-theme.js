const fs = require('fs');
const path = require('path');

function processDir(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const fullPath = path.join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
            processDir(fullPath);
        } else if (fullPath.endsWith('.tsx') || fullPath.endsWith('.ts')) {
            let content = fs.readFileSync(fullPath, 'utf8');
            let original = content;

            // Update old blue shadows to cyan shadows
            content = content.replace(/rgba\(0,127,255/g, 'rgba(0,242,255');

            // Update solid primary buttons to gradient buttons
            content = content.replace(/bg-primary text-primary-foreground/g, 'bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white border-none');

            // Update old text gradients
            content = content.replace(/from-primary to-cyan-400/g, 'from-[#7000ff] to-[#00f2ff]');
            
            // Fix any old button drop shadows that used old blue hex
            content = content.replace(/shadow-\[0_0_([0-9]+)px_#006FEE\]/g, 'shadow-[0_0_$1px_rgba(0,242,255,0.4)]');

            if (content !== original) {
                fs.writeFileSync(fullPath, content, 'utf8');
                console.log('Updated:', fullPath);
            }
        }
    }
}

processDir('/Users/erebus2507/Desktop/BudAI---Personal-Financing-AI-Agent-and-Analyser/budai-frontend/app');
