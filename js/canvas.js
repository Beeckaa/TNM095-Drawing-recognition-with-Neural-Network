var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');

var radius = 2;
var start = 0;
var end = Math.PI * 2;
var dragging = false;

canvas.width = 600;
canvas.height = 400;

context.lineWidth = radius * 2;

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

async function predict() {

	canvas.toBlob(function(blob) {
		image = blob;
		newImg = document.createElement('img'),
		url = URL.createObjectURL(blob);
	
		newImg.onload = function() {
		// no longer need to read the blob so it's revoked
		URL.revokeObjectURL(url);
		  };
	
		newImg.src = url;
		document.body.appendChild(newImg);
	});

	var predictions = await net.predict(preprocessCanvas(canvas)).data(); // this will run whenever "tensor" updates
	console.log(predictions);
	console.log(tf.argMax(predictions).data());
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

preprocessCanvas = (drawing) => {
	// Preprocess image for the network
	console.log(tf.browser.fromPixels(drawing));
	let tensor = tf
	.browser.fromPixels(drawing) // Shape: (600, 400, 3) - RGB image
	.resizeNearestNeighbor([28, 28]) // Shape: (28, 28, 3) - RGB image
	.mean(2) // Shape: (28, 28) - grayscale
	.flatten()
	//.expandDims(2) // Shape: (28, 28, 1) - network expects 3d values with channels in the last dimension
	.expandDims() // Shape: (1, 28, 28) - network makes predictions for "batches" of images
	.toFloat(); // Network works with floating points inputs
	return tensor.div(255.0); // Normalize [0..255] values into [0..1] range
}

