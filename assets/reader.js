const $ = require('jquery');

MathJax.Hub.Config({
	tex2jax: {
		inlineMath: [['$','$']],
		displayMath: [['$$','$$']],
	}
});

$(document).ready(() => {
	$('.equation').each((index, element) => {
		const tex = $(element).text();

		if ($(element).is('div')) {
			$(element).text(`$$${tex}$$`);
		} else {
			$(element).text(`$${tex}$`);
		}

		MathJax.Hub.Queue(['Typeset', MathJax.Hub, element]);
	});
});
