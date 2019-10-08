var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var i;
var radius = 2;
var start = 0;
var end = Math.PI * 2;
var dragging = false;

canvas.width = 600;
canvas.height = 400;

context.lineWidth = radius * 2;

let label;
let net;
var predictions;

async function app() {
    console.log('Loading model..');

    // Load the model.
    net = await tf.loadLayersModel('http://localhost:8080/model.json');
	console.log('Successfully loaded model');
}

app();

var clearCanvas = function() {
    context.clearRect(0, 0, canvas.width, canvas.height);
}

var drawing = function(e){
	if(dragging) {
		context.strokeStyle = '#404040';
		context.fillStyle = '#404040';
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
	context.save()
}

async function predict() { 
	imgData = context.getImageData(0,0, 600, 400);

	//context.putImageData(imgData, 100, 100);

	var predictions = await net.predict(preprocessCanvas(imgData)); // this will run whenever "tensor" updates
	console.log("Predictions: " + predictions);

	a = predictions.dataSync();
	arr = Array.from(a);

	console.log(arr)

	label = arr.indexOf(Math.max(...arr));

	switch(label) {
		case 0:
			console.log('Bird')
			break;
		case 1:
			console.log('Sheep')
			break;
		case 2:
			console.log('Turtle')
			break;
		case 3:
			console.log('Hedgehog')
			break;
		case 4:
			console.log('Octopus')
			break;
		case 5:
			console.log('Giraffe')
			break;
		default:
			break;
		}

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
		context.strokeStyle = '#404040';
		context.fillStyle = '#404040';
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

preprocessCanvas = (drawing) => {
	// Preprocess image for the network
	let tensor = tf
	.browser.fromPixels(drawing) // Shape: (600, 400, 3) - RGB image
	.resizeNearestNeighbor([28, 28], align_corners=true) // Shape: (28, 28, 3) - RGB image
	.mean(2) // Shape: (28, 28) - grayscale
	.flatten() // Shape: (784)
	.expandDims() // Shape: (1, 784) - network makes predictions for "batches" of images
	.toFloat(); // Network works with floating points inputs
	return tensor.div(255.0); // Normalize [0..255] values into [0..1] range
}

