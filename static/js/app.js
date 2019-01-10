var App = function() {
}


$(function() {

	// ajaxローディング設定
	$(document).ajaxStart(function() {
		$("#loading").fadeIn("fast");
	});
	$(document).ajaxStop(function() {
		$("#loading").fadeOut("fast");
	});
});  
