window.scrollTo(0,document.body.scrollHeight);

scrollDownAfter500ms().then(() => {
	return scrollDownAfter500ms();
}).then(() => {
	return clickLoadMoreEvery2Sec();
}).then(() => {
	alert("loaded all the photos");
	downloadAllImages();
});

function scrollDownAfter500ms() {
	return new Promise(resolve => {
		setTimeout(() => {
			window.scrollTo(0, document.body.scrollHeight);
			resolve();
		}, 1500);
	})
}

function clickLoadMoreEvery2Sec() {
	return new Promise(resolve => {
		const interval = setInterval(() => {
			let loadMoreButton = $("button.photos__load-more");

			if(!loadMoreButton.hasClass("hidden")){
				loadMoreButton.click();
				window.scrollTo(0, document.body.scrollHeight);
			}else{
				clearInterval(interval);
				resolve();
			}
		}, 3000)
	})
}

function downloadAllImages() {
const regexToGetFullSizeUriLink = /FullSizeUri":"(.*)","SmallThumbUri/;

$("div.page-item > a > img.l-img-responsive").each((index, element) => {
	let fullImageUri = regexToGetFullSizeUriLink.exec(element.parentElement.attributes['ng-click'].value)[1];

	let downloadElement = $("<a>")
		.attr("href", fullImageUri)
		.attr("download", "sheker.png")
		.appendTo("body");

	downloadElement[0].click();

	downloadElement.remove();
});
}