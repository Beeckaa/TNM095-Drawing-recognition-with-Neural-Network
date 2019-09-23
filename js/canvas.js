var canvas = document.getElementById('canvas');
var statusdiv = document.getElementById('statusdiv')
var context = canvas.getContext('2d');

var radius = 2;
var start = 0;
var end = Math.PI * 2;
var dragging = false;

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

context.lineWidth = radius * 2;

var drawing = function(e){
	if(dragging) {
		context.lineTo(e.offsetX, e.offsetY);
		context.stroke();
		context.beginPath();
		context.arc(e.offsetX, e.offsetY, radius, start, end);
		context.fill();
		context.beginPath();
		context.moveTo(e.offsetX, e.offsetY);
	}
}

var startDraw = function(e){
	dragging = true;
	drawing(e);
}

var stopDraw = function(){
	dragging = false;
	context.beginPath();
}

// EventListeners
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', drawing);
canvas.addEventListener('mouseup', stopDraw);

// Eventlistener and function for touchstart
canvas.addEventListener('touchstart', function(e){
    dragging = true;
    e.preventDefault()
}, false);

// Eventlistener and function for touchmove
canvas.addEventListener('touchmove', function(e){
    var touchobj = e.changedTouches[0] // Reference first touch point for this event
    if(dragging) {
		context.lineTo(touchobj.clientX, touchobj.clientY);
		context.stroke();
		context.beginPath();
		context.arc(touchobj.clientX, touchobj.clientY, radius, start, end);
		context.fill();
		context.beginPath();
		context.moveTo(touchobj.clientX, touchobj.clientY);
	}
    e.preventDefault()
}, false);

// Eventlistener and function for touchend
canvas.addEventListener('touchend', function(e){
    dragging = false;
	context.beginPath();
    e.preventDefault()
}, false);

