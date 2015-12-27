const $ = require('jquery');
const katex = require('katex');

$(document).ready(() => {
	$('.equation').each((index, element) => {
		const tex = $(element).text();
		katex.render(tex, element);
	});
});
