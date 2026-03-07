// Make notebook text outputs scrollable with an expand/collapse toggle
document.addEventListener('DOMContentLoaded', function() {
    var MAX_HEIGHT = 300; // px — must match the CSS max-height value

    // Selectors for text output pre elements inside nbsphinx output blocks
    var preSelectors = [
        '.nboutput pre',
        '.nboutput .highlight pre',
        '.nboutput .output_subarea.output_stream pre',
        '.nboutput .output_subarea.output_text pre',
    ];

    preSelectors.forEach(function(selector) {
        document.querySelectorAll(selector).forEach(function(pre) {
            // Only add controls if content actually overflows
            if (pre.scrollHeight <= MAX_HEIGHT) return;

            // Wrap the pre in a relative container so we can position the button
            var wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(pre);

            // Create a fade overlay + toggle button
            var overlay = document.createElement('div');
            overlay.className = 'nb-output-overlay';
            overlay.style.cssText = [
                'position:absolute', 'bottom:0', 'left:0', 'right:0', 'height:40px',
                'background:linear-gradient(transparent,var(--color-background-primary,#fff))',
                'pointer-events:none',
            ].join(';');
            wrapper.appendChild(overlay);

            var btn = document.createElement('button');
            btn.textContent = 'Show more';
            btn.className = 'nb-output-toggle';
            btn.style.cssText = [
                'display:block', 'margin:2px auto 0', 'padding:2px 10px',
                'font-size:0.8em', 'cursor:pointer',
                'border:1px solid #aaa', 'border-radius:3px',
                'background:#f8f9fa', 'color:#444',
            ].join(';');

            var expanded = false;
            btn.addEventListener('click', function() {
                expanded = !expanded;
                if (expanded) {
                    pre.style.maxHeight = 'none';
                    overlay.style.display = 'none';
                    btn.textContent = 'Show less';
                } else {
                    pre.style.maxHeight = MAX_HEIGHT + 'px';
                    overlay.style.display = '';
                    btn.textContent = 'Show more';
                    // Scroll back to top of pre so user sees the beginning
                    pre.scrollTop = 0;
                }
            });

            wrapper.appendChild(btn);
        });
    });
});
