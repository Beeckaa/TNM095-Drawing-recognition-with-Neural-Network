var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');

var radius = 2;
var start = 0;
var end = Math.PI * 2;
var dragging = false;

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

context.lineWidth = radius * 2;

var drawing = function(e){
	if(dragging){
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