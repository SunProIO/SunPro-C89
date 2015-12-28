const $ = require('jquery');
const url = require('url');

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

$.get('../assets/catalog.json', (articles, status) => {
	const href = url.parse(window.location.href);
	const paths = href.pathname.split('/')
	const slug = paths[paths.length - 2];

	const article = articles.filter(it => it.slug === slug)[0];

	// Error if article is not found
	if (article === undefined) {
		return console.error('loading article data failed');
	}

	$(document).ready(() => {
		$('header h1 > a').text(`${article.title} ${article.author}`);
	});
});
